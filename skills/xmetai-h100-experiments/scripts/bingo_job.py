#!/usr/bin/env python3
"""Submit and inspect BingoCloud Kubernetes training jobs without a browser."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import getpass
import http.cookiejar
import json
import os
import re
import shlex
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

DEFAULT_BASE_URL = ""
DEFAULT_DEX_URL = ""
DEFAULT_CLUSTER_ID = ""
DEFAULT_WORKSPACE_ID = ""
DEFAULT_NAMESPACE = "default"
DEFAULT_USERNAME = ""
DEFAULT_CLIENT_ID = "kube"
DEFAULT_PROXY_URL: Optional[str] = None
DNS_LABEL_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


class PlatformError(RuntimeError):
    """A platform authentication, API, or YAML validation error."""


class PlatformHttpError(PlatformError):
    """An HTTP error returned by the platform."""

    def __init__(self, status: int, method: str, url: str, message: str) -> None:
        super().__init__(f"{method} {url}: HTTP {status}: {message}")
        self.status = status


class RedirectRecorder(urllib.request.HTTPRedirectHandler):
    """Record OAuth redirects while retaining urllib's normal redirect behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.locations: List[str] = []

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        self.locations.append(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class ForceProxyHandler(urllib.request.ProxyHandler):
    """Use the configured proxy even when macOS marks a private IP as bypassed."""

    def proxy_open(self, req: urllib.request.Request, proxy: str, proxy_type: str) -> Optional[urllib.response.addinfourl]:
        original_type = req.type
        parsed_type, user, password, hostport = urllib.request._parse_proxy(proxy)
        if parsed_type is None:
            parsed_type = original_type
        if user and password:
            credentials = f"{urllib.parse.unquote(user)}:{urllib.parse.unquote(password)}"
            encoded = base64.b64encode(credentials.encode()).decode("ascii")
            req.add_header("Proxy-Authorization", f"Basic {encoded}")
        req.set_proxy(urllib.parse.unquote(hostport), parsed_type)
        if original_type == parsed_type or original_type == "https":
            return None
        return self.parent.open(req, timeout=req.timeout)


def _json_error_message(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return "empty response"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    if isinstance(payload, Mapping):
        return str(payload.get("message") or payload.get("msg") or payload.get("error_description") or payload.get("error") or payload)[:500]
    return str(payload)[:500]


def _url_params(url: str) -> Dict[str, List[str]]:
    parts = urllib.parse.urlsplit(url)
    params = urllib.parse.parse_qs(parts.query)
    fragment = parts.fragment.split("?", 1)[1] if "?" in parts.fragment else parts.fragment
    params.update(urllib.parse.parse_qs(fragment))
    return params


def _jwt_expiry(token: str) -> Optional[float]:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload.encode()))
        return float(decoded["exp"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class BingoClient:
    """Minimal client for the platform's Dex SSO and Kubernetes proxy APIs."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        dex_url: str = DEFAULT_DEX_URL,
        cluster_id: str = DEFAULT_CLUSTER_ID,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        username: str = DEFAULT_USERNAME,
        password: Optional[str] = None,
        proxy_url: Optional[str] = DEFAULT_PROXY_URL,
        verify_tls: bool = False,
        use_token_cache: bool = True,
        token_cache_path: Optional[Path] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.dex_url = dex_url.rstrip("/")
        self.cluster_id = cluster_id
        self.workspace_id = workspace_id
        self.username = username
        self.password = password
        self.proxy_url = proxy_url
        self.use_token_cache = use_token_cache
        self.token_cache_path = token_cache_path or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "xmetai" / "bingo_job_token.json"
        self._token: Optional[str] = None
        self._token_was_cached = False
        self._cookie_jar = http.cookiejar.CookieJar()
        self._redirects = RedirectRecorder()
        ssl_context = ssl.create_default_context() if verify_tls else ssl._create_unverified_context()
        proxy_handler = ForceProxyHandler({"http": proxy_url, "https": proxy_url}) if proxy_url else urllib.request.ProxyHandler({})
        self._opener = urllib.request.build_opener(
            proxy_handler,
            urllib.request.HTTPSHandler(context=ssl_context),
            urllib.request.HTTPCookieProcessor(self._cookie_jar),
            self._redirects,
        )

    @property
    def redirect_uri(self) -> str:
        return f"{self.base_url}/kubeverse/#/ssoclient?returnUrl=/"

    def _open(
        self,
        url: str,
        *,
        method: str = "GET",
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> urllib.response.addinfourl:
        request_headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "xmetai-bingo-job/1",
        }
        request_headers.update(headers or {})
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            return self._opener.open(request, timeout=30)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            raise PlatformHttpError(exc.code, method, urllib.parse.urlsplit(url).path, _json_error_message(body)) from exc
        except urllib.error.URLError as exc:
            raise PlatformError(f"{method} {urllib.parse.urlsplit(url).path}: {exc.reason}") from exc

    def _read_json(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: Optional[Mapping[str, Any]] = None,
        form: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        request_headers = dict(headers or {})
        data: Optional[bytes] = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        elif form is not None:
            data = urllib.parse.urlencode(form).encode("utf-8")
            request_headers["Content-Type"] = "application/x-www-form-urlencoded"
        with self._open(url, method=method, data=data, headers=request_headers) as response:
            body = response.read()
        if not body:
            return {}
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise PlatformError(f"{method} {urllib.parse.urlsplit(url).path}: expected JSON response") from exc
        if not isinstance(result, dict):
            raise PlatformError(f"{method} {urllib.parse.urlsplit(url).path}: expected a JSON object")
        return result

    def _load_cached_token(self) -> Optional[str]:
        if not self.use_token_cache or not self.token_cache_path.exists():
            return None
        try:
            cache = json.loads(self.token_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if cache.get("base_url") != self.base_url or cache.get("username") != self.username:
            return None
        if float(cache.get("expires_at", 0)) <= time.time() + 30:
            return None
        token = cache.get("access_token")
        return token if isinstance(token, str) and token else None

    def _write_cached_token(self, token: str, expires_at: float) -> None:
        if not self.use_token_cache:
            return
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.token_cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "base_url": self.base_url,
                    "username": self.username,
                    "access_token": token,
                    "expires_at": expires_at,
                }
            ),
            encoding="utf-8",
        )
        os.chmod(temporary_path, 0o600)
        temporary_path.replace(self.token_cache_path)

    def _clear_cached_token(self) -> None:
        self._token = None
        self._token_was_cached = False
        if self.use_token_cache:
            try:
                self.token_cache_path.unlink()
            except FileNotFoundError:
                pass

    def _login(self) -> str:
        self._redirects.locations.clear()
        oauth_query = urllib.parse.urlencode(
            {
                "client_id": DEFAULT_CLIENT_ID,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": "openid profile email offline_access",
            }
        )
        auth_url = f"{self.dex_url}/dex/auth?{oauth_query}"
        dex_headers = {
            "Origin": self.dex_url,
            "Referer": auth_url,
            "X-Requested-With": "XMLHttpRequest",
        }
        with self._open(auth_url, headers=dex_headers) as response:
            response.read()

        connectors = self._read_json(f"{self.dex_url}/dex/connectors?{oauth_query}", headers=dex_headers)
        connector_items = connectors.get("data")
        if not isinstance(connector_items, list):
            raise PlatformError("Dex did not return a connector list")
        connector = next((item for item in connector_items if item.get("id") == "local"), None)
        if not connector:
            raise PlatformError("Dex local password connector is unavailable")

        connector_url = urllib.parse.urljoin(self.dex_url + "/", str(connector["connUrl"]))
        with self._open(connector_url, headers=dex_headers) as response:
            response.read()

        password = self.password or os.environ.get("BINGO_PASSWORD")
        if not password:
            password = getpass.getpass(f"BingoCloud password for {self.username}: ")

        prelogin_url = urllib.parse.urljoin(self.dex_url + "/", str(connector["preUrl"]))
        prelogin = self._read_json(
            prelogin_url,
            method="POST",
            payload={"login": self.username, "password": password, "code": ""},
            headers=dex_headers,
        )
        sign = (prelogin.get("data") or {}).get("sign")
        if not sign:
            raise PlatformError(str(prelogin.get("msg") or prelogin.get("message") or "BingoCloud login failed"))

        login_url = urllib.parse.urljoin(self.dex_url + "/", str(connector["url"]))
        login_form = urllib.parse.urlencode({"login": self.username, "password": password, "sign": sign}).encode("utf-8")
        with self._open(
            login_url,
            method="POST",
            data=login_form,
            headers={**dex_headers, "Content-Type": "application/x-www-form-urlencoded"},
        ) as response:
            final_url = response.geturl()
            response.read()

        code: Optional[str] = None
        state: Optional[str] = None
        for location in self._redirects.locations + [final_url]:
            params = _url_params(location)
            code = code or (params.get("code") or [None])[0]
            state = state or (params.get("state") or [None])[0]
        if not code:
            raise PlatformError("Dex login did not return an authorization code")

        exchange_payload: Dict[str, Any] = {"code": code, "redirectUrl": self.redirect_uri}
        if state:
            exchange_payload["state"] = state
        exchange = self._read_json(f"{self.base_url}/api/auth/ssoLogin", method="POST", payload=exchange_payload)
        exchange_data = exchange.get("data") or {}
        token = exchange_data.get("accessToken")
        if not token:
            raise PlatformError(str(exchange.get("msg") or exchange.get("message") or "BingoCloud SSO exchange failed"))

        expires_at = _jwt_expiry(token)
        if expires_at is None:
            expires_in = float(exchange_data.get("expireIn") or 3600)
            expires_at = time.time() + expires_in
        self._write_cached_token(token, expires_at)
        return token

    def access_token(self) -> str:
        if self._token:
            return self._token
        cached = self._load_cached_token()
        if cached:
            self._token = cached
            self._token_was_cached = True
            return cached
        self._token = self._login()
        self._token_was_cached = False
        return self._token

    def api_request(
        self,
        path: str,
        *,
        method: str = "GET",
        query: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({key: value for key, value in query.items() if value is not None})
        for attempt in range(2):
            token = self.access_token()
            try:
                return self._read_json(url, method=method, payload=payload, headers={"Authorization": f"Bearer {token}"})
            except PlatformHttpError as exc:
                if exc.status != 401 or attempt == 1:
                    raise
                self._clear_cached_token()
        raise AssertionError("unreachable")

    def k8s_path(self, suffix: str) -> str:
        return f"/verse-apis/v1/k8s/{urllib.parse.quote(self.cluster_id)}/{urllib.parse.quote(self.workspace_id)}/{suffix.lstrip('/')}"

    def list_jobs(self, *, namespace: str, keyword: str = "", page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        return self.api_request(
            self.k8s_path("jobs"),
            query=_list_query(namespace=namespace, keyword=keyword, page=page, page_size=page_size),
        )

    def list_pods(self, *, namespace: str, keyword: str = "", page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        return self.api_request(
            self.k8s_path("pods"),
            query=_list_query(namespace=namespace, keyword=keyword, page=page, page_size=page_size),
        )

    def delete_pod(self, *, namespace: str, name: str) -> Dict[str, Any]:
        suffix = f"pod/{urllib.parse.quote(namespace)}/{urllib.parse.quote(name)}"
        return self.api_request(self.k8s_path(suffix), method="DELETE")

    def job_detail(self, *, namespace: str, name: str) -> Dict[str, Any]:
        suffix = f"job/{urllib.parse.quote(namespace)}/{urllib.parse.quote(name)}"
        return self.api_request(self.k8s_path(suffix))

    def list_events(
        self,
        *,
        namespace: str,
        name: str,
        creation_timestamp: str,
        kind: str = "Job",
        page_size: int = 50,
    ) -> Dict[str, Any]:
        return self.api_request(
            self.k8s_path("events"),
            query={
                "page": 1,
                "pageSize": page_size,
                "name": name,
                "namespace": namespace,
                "kind": kind,
                "creationTimestamp": creation_timestamp,
                "keyword": "",
                "orderBy": "lastTimestamp",
                "ascending": "false",
            },
        )

    def submit_yaml(self, *, namespace: str, yaml_text: str) -> Dict[str, Any]:
        return self.api_request(
            self.k8s_path("yaml"),
            method="POST",
            payload={"namespace": namespace, "yamlStr": yaml_text, "type": "create"},
        )


def _list_query(*, namespace: str, keyword: str, page: int, page_size: int) -> Dict[str, Any]:
    return {
        "page": page,
        "pageSize": page_size,
        "keyword": keyword,
        "namespace": namespace,
        "orderBy": "creationTimestamp",
        "ascending": "false",
    }


@dataclasses.dataclass
class JobOverrides:
    name: Optional[str] = None
    namespace: Optional[str] = None
    container: Optional[str] = None
    shell_command: Optional[str] = None
    command_json: Optional[str] = None
    args_json: Optional[str] = None
    cpu: Optional[str] = None
    memory: Optional[str] = None
    gpus: Optional[str] = None
    vgpu_cores: Optional[str] = None
    vgpu_mem: Optional[str] = None


def _validate_dns_label(value: str, field: str) -> None:
    if len(value) > 63 or not DNS_LABEL_RE.fullmatch(value):
        raise PlatformError(f"{field} must be a valid lowercase Kubernetes DNS label (maximum 63 characters): {value!r}")


def _json_string_list(value: str, option: str) -> List[str]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PlatformError(f"{option} must be a JSON array of strings") from exc
    if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
        raise PlatformError(f"{option} must be a JSON array of strings")
    return result


def _container_tokens(container: Mapping[str, Any]) -> List[str]:
    command = container.get("command")
    args = container.get("args")
    command_tokens = [str(value) for value in command] if isinstance(command, list) else []
    arg_tokens = [str(value) for value in args] if isinstance(args, list) else []
    if len(arg_tokens) == 1 and command_tokens and command_tokens[-1] in {"-c", "-lc"}:
        try:
            arg_tokens = shlex.split(arg_tokens[0])
        except ValueError as exc:
            raise PlatformError(f"cannot parse --shell-command while deriving the Job name: {exc}") from exc
    return command_tokens + arg_tokens


def _option_value(tokens: Sequence[str], flags: Sequence[str]) -> Optional[str]:
    for index, token in enumerate(tokens[:-1]):
        if token in flags:
            return tokens[index + 1]
    for token in tokens:
        for flag in flags:
            if token.startswith(f"{flag}="):
                return token.split("=", 1)[1]
    return None


def _default_job_name(container: Mapping[str, Any], today: Optional[dt.date] = None) -> str:
    tokens = _container_tokens(container)

    config_path = _option_value(tokens, ("-m", "--model", "--config"))
    if config_path is None:
        config_path = next((token for token in tokens if token.endswith(".py") and "config" in token.lower()), None)
    if not config_path:
        raise PlatformError("cannot derive the default Job name: no config .py was found in container command/args; pass --name")

    stage = (_option_value(tokens, ("-s", "--stage")) or "train").lower()
    stage = {"ltrain": "train", "leval": "eval"}.get(stage, stage)
    if stage not in {"train", "eval", "export"}:
        raise PlatformError(f"cannot derive the default Job name from unsupported stage {stage!r}; pass --name")

    config_name = re.sub(r"[^a-z0-9]+", "-", Path(config_path).stem.lower()).strip("-")
    if not config_name:
        raise PlatformError(f"cannot derive a valid Job name from config path {config_path!r}; pass --name")
    date_text = (today or dt.datetime.now().astimezone().date()).strftime("%Y%m%d")
    prefix = f"{stage}-"
    suffix = f"-{date_text}"
    config_name = config_name[: 63 - len(prefix) - len(suffix)].rstrip("-")
    return f"{prefix}{config_name}{suffix}"


def patch_job_documents(documents: List[Dict[str, Any]], overrides: JobOverrides) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    jobs = [document for document in documents if isinstance(document, dict) and document.get("kind") == "Job"]
    if len(jobs) != 1:
        raise PlatformError(f"expected exactly one Job document, found {len(jobs)}")
    job = jobs[0]

    metadata = job.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise PlatformError("Job metadata must be a mapping")
    namespace = overrides.namespace or metadata.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise PlatformError("Job metadata.namespace is required")
    _validate_dns_label(namespace, "namespace")
    metadata["namespace"] = namespace

    try:
        template = job["spec"]["template"]
        pod_spec = template["spec"]
        containers = pod_spec["containers"]
    except (KeyError, TypeError) as exc:
        raise PlatformError("Job must define spec.template.spec.containers") from exc
    if not isinstance(containers, list) or not containers:
        raise PlatformError("Job must contain at least one container")

    template_metadata = template.setdefault("metadata", {})
    labels = template_metadata.setdefault("labels", {})
    if not isinstance(labels, dict):
        raise PlatformError("Job spec.template.metadata.labels must be a mapping")

    container: Optional[Dict[str, Any]] = None
    if overrides.container:
        container = next((item for item in containers if isinstance(item, dict) and item.get("name") == overrides.container), None)
        if container is None:
            raise PlatformError(f"container {overrides.container!r} was not found")
    elif isinstance(containers[0], dict):
        container = containers[0]
    if container is None:
        raise PlatformError("selected container must be a mapping")

    command_options = sum(value is not None for value in (overrides.shell_command, overrides.command_json))
    if command_options > 1:
        raise PlatformError("--shell-command and --command-json are mutually exclusive")
    if overrides.shell_command is not None and overrides.args_json is not None:
        raise PlatformError("--shell-command and --args-json are mutually exclusive")
    if overrides.shell_command is not None:
        container["command"] = ["bash", "-lc"]
        container["args"] = [overrides.shell_command]
    else:
        if overrides.command_json is not None:
            container["command"] = _json_string_list(overrides.command_json, "--command-json")
        if overrides.args_json is not None:
            container["args"] = _json_string_list(overrides.args_json, "--args-json")
        if overrides.gpus is not None and isinstance(container.get("args"), list):
            for index, argument in enumerate(container["args"][:-1]):
                if argument in {"-g", "--gpus"}:
                    container["args"][index + 1] = str(overrides.gpus)
                    break

    name = overrides.name or _default_job_name(container)
    _validate_dns_label(name, "Job name")
    metadata["name"] = name
    labels["job-name"] = name

    resources = container.setdefault("resources", {})
    if not isinstance(resources, dict):
        raise PlatformError("selected container resources must be a mapping")
    requests = resources.setdefault("requests", {})
    limits = resources.setdefault("limits", {})
    if not isinstance(requests, dict) or not isinstance(limits, dict):
        raise PlatformError("selected container resource requests and limits must be mappings")
    resource_overrides = {
        "cpu": overrides.cpu,
        "memory": overrides.memory,
        "nvidia.com/vgpu": overrides.gpus,
        "nvidia.com/vgpu-cores": overrides.vgpu_cores,
        "nvidia.com/vgpu-mem": overrides.vgpu_mem,
    }
    for key, value in resource_overrides.items():
        if value is not None:
            requests[key] = str(value)
            limits[key] = str(value)

    return documents, {"name": name, "namespace": namespace, "container": container.get("name", "")}


def render_job_yaml(template_path: Path, overrides: JobOverrides) -> Tuple[str, Dict[str, Any]]:
    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlatformError(f"cannot read YAML template {template_path}: {exc}") from exc
    try:
        raw_documents = list(yaml.safe_load_all(template_text))
    except yaml.YAMLError as exc:
        raise PlatformError(f"invalid YAML template {template_path}: {exc}") from exc
    documents = [document for document in raw_documents if document is not None]
    patched_documents, metadata = patch_job_documents(documents, overrides)
    yaml_text = yaml.safe_dump_all(patched_documents, allow_unicode=True, sort_keys=False)
    return yaml_text, metadata


def _response_data(response: Mapping[str, Any]) -> Any:
    return response.get("data", response)


def _items(response: Mapping[str, Any]) -> List[Dict[str, Any]]:
    data = _response_data(response)
    if isinstance(data, Mapping) and isinstance(data.get("items"), list):
        return [item for item in data["items"] if isinstance(item, dict)]
    return []


def _object(item: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("Object", "object", "Job", "Pod"):
        value = item.get(key)
        if isinstance(value, Mapping):
            return value
    data = item.get("data")
    if isinstance(data, Mapping):
        return _object(data)
    return item


def _metadata(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _object(item).get("metadata")
    return value if isinstance(value, Mapping) else {}


def _status_mapping(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _object(item).get("status")
    return value if isinstance(value, Mapping) else {}


def _job_status(item: Mapping[str, Any]) -> str:
    status = _status_mapping(item)
    for condition in status.get("conditions") or []:
        if isinstance(condition, Mapping) and str(condition.get("status")).lower() == "true" and condition.get("type") in {"Complete", "Failed"}:
            return str(condition["type"])
    if status.get("active"):
        return "Running"
    if status.get("succeeded") and not status.get("active"):
        return "Succeeded"
    if status.get("failed"):
        return "Failed"
    platform_status = item.get("resourceStatus")
    if platform_status:
        return str(platform_status)
    return "Pending"


def _pod_status(item: Mapping[str, Any]) -> str:
    status = _status_mapping(item)
    for container_status in status.get("containerStatuses") or []:
        state = container_status.get("state") if isinstance(container_status, Mapping) else None
        if not isinstance(state, Mapping):
            continue
        waiting = state.get("waiting")
        terminated = state.get("terminated")
        if isinstance(waiting, Mapping) and waiting.get("reason"):
            return str(waiting["reason"])
        if isinstance(terminated, Mapping) and terminated.get("reason"):
            return str(terminated["reason"])
    return str(status.get("phase") or item.get("resourceStatus") or "Unknown")


def _created_at(item: Mapping[str, Any]) -> str:
    return str(item.get("CreationTimestamp") or _metadata(item).get("creationTimestamp") or "-")


def _job_name_for_pod(item: Mapping[str, Any]) -> str:
    metadata = _metadata(item)
    labels = metadata.get("labels")
    if isinstance(labels, Mapping) and labels.get("job-name"):
        return str(labels["job-name"])
    for reference in metadata.get("ownerReferences") or []:
        if isinstance(reference, Mapping) and reference.get("kind") == "Job":
            return str(reference.get("name") or "")
    return ""


def _job_row(item: Mapping[str, Any]) -> Dict[str, Any]:
    obj = _object(item)
    spec = obj.get("spec") if isinstance(obj.get("spec"), Mapping) else {}
    status = _status_mapping(item)
    metadata = _metadata(item)
    return {
        "name": item.get("Name") or metadata.get("name") or "-",
        "namespace": metadata.get("namespace") or "-",
        "status": _job_status(item),
        "active": status.get("active", 0),
        "succeeded": status.get("succeeded", 0),
        "failed": status.get("failed", 0),
        "parallelism": spec.get("parallelism", "-"),
        "created": _created_at(item),
    }


def _pod_row(item: Mapping[str, Any]) -> Dict[str, Any]:
    obj = _object(item)
    spec = obj.get("spec") if isinstance(obj.get("spec"), Mapping) else {}
    metadata = _metadata(item)
    return {
        "name": item.get("Name") or metadata.get("name") or "-",
        "namespace": metadata.get("namespace") or "-",
        "status": _pod_status(item),
        "job": _job_name_for_pod(item) or "-",
        "node": spec.get("nodeName") or "-",
        "created": _created_at(item),
    }


def _pod_scheduled_condition(item: Mapping[str, Any]) -> Mapping[str, Any]:
    for condition in _status_mapping(item).get("conditions") or []:
        if isinstance(condition, Mapping) and condition.get("type") == "PodScheduled":
            return condition
    return {}


def _is_queued_pod(item: Mapping[str, Any]) -> bool:
    if not _job_name_for_pod(item):
        return False
    status = _status_mapping(item)
    phase = status.get("phase") or item.get("phase") or item.get("resourceStatus")
    if str(phase).lower() != "pending":
        return False
    scheduled = _pod_scheduled_condition(item)
    if scheduled:
        return str(scheduled.get("status")).lower() != "true"
    obj = _object(item)
    spec = obj.get("spec") if isinstance(obj.get("spec"), Mapping) else {}
    return not spec.get("nodeName")


def _queued_pods(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((item for item in items if _is_queued_pod(item)), key=lambda item: (_created_at(item), str(_metadata(item).get("name") or "")))


def _queue_row(item: Mapping[str, Any]) -> Dict[str, Any]:
    scheduled = _pod_scheduled_condition(item)
    return {
        "pod": item.get("Name") or _metadata(item).get("name") or "-",
        "job": _job_name_for_pod(item) or "-",
        "reason": scheduled.get("reason") or "PendingScheduling",
        "message": scheduled.get("message") or "waiting for a scheduler decision",
        "created": _created_at(item),
    }


def _event_row(item: Mapping[str, Any]) -> Dict[str, Any]:
    involved = item.get("involvedObject") if isinstance(item.get("involvedObject"), Mapping) else {}
    source = item.get("source") if isinstance(item.get("source"), Mapping) else {}
    return {
        "last": item.get("lastTimestamp") or _metadata(item).get("creationTimestamp") or "-",
        "type": item.get("type") or "-",
        "reason": item.get("reason") or "-",
        "object": f"{involved.get('kind', '-')}/{involved.get('name', '-')}",
        "component": source.get("component") or "-",
        "message": item.get("message") or "-",
    }


def _trim(value: Any, limit: int = 80) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def print_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[Tuple[str, str]], *, file: Any = None) -> None:
    output = file or sys.stdout
    if not rows:
        print("(no results)", file=output)
        return
    widths: List[int] = []
    for key, title in columns:
        widths.append(max(len(title), *(len(_trim(row.get(key, "-"))) for row in rows)))
    print("  ".join(title.ljust(width) for (_, title), width in zip(columns, widths)), file=output)
    print("  ".join("-" * width for width in widths), file=output)
    for row in rows:
        print("  ".join(_trim(row.get(key, "-")).ljust(width) for (key, _), width in zip(columns, widths)), file=output)


def _build_client(args: argparse.Namespace) -> BingoClient:
    required = {
        "--base-url/BINGO_BASE_URL": args.base_url,
        "--dex-url/BINGO_DEX_URL": args.dex_url,
        "--cluster-id/BINGO_CLUSTER_ID": args.cluster_id,
        "--workspace-id/BINGO_WORKSPACE_ID": args.workspace_id,
        "--username/BINGO_USERNAME": args.username,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise PlatformError(f"missing required platform settings: {', '.join(missing)}")
    return BingoClient(
        base_url=args.base_url,
        dex_url=args.dex_url,
        cluster_id=args.cluster_id,
        workspace_id=args.workspace_id,
        username=args.username,
        proxy_url=None if args.no_proxy else args.proxy_url,
        verify_tls=args.verify_tls,
        use_token_cache=not args.no_token_cache,
    )


def _overrides_from_args(args: argparse.Namespace) -> JobOverrides:
    return JobOverrides(
        name=args.name,
        namespace=args.namespace,
        container=args.container,
        shell_command=args.shell_command,
        command_json=args.command_json,
        args_json=args.args_json,
        cpu=args.cpu,
        memory=args.memory,
        gpus=args.gpus,
        vgpu_cores=args.vgpu_cores,
        vgpu_mem=args.vgpu_mem,
    )


def _write_rendered_yaml(yaml_text: str, output: Optional[Path]) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml_text, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        sys.stdout.write(yaml_text)


def command_render(args: argparse.Namespace) -> int:
    yaml_text, _ = render_job_yaml(args.template, _overrides_from_args(args))
    _write_rendered_yaml(yaml_text, args.output)
    return 0


def command_submit(args: argparse.Namespace) -> int:
    yaml_text, metadata = render_job_yaml(args.template, _overrides_from_args(args))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml_text, encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.dry_run:
        if not args.output:
            sys.stdout.write(yaml_text)
        print(f"Dry run only; did not submit Job {metadata['namespace']}/{metadata['name']}", file=sys.stderr)
        return 0
    client = _build_client(args)
    if not args.allow_queued:
        queued = _queued_pods(_list_all_pods(client, namespace=metadata["namespace"], keyword="", page_size=100))
        if queued:
            print("Queued Job Pods already exist:", file=sys.stderr)
            print_table(
                [_queue_row(item) for item in queued],
                [("pod", "POD"), ("job", "JOB"), ("reason", "REASON"), ("message", "MESSAGE"), ("created", "CREATED")],
                file=sys.stderr,
            )
            raise PlatformError(f"refusing to submit while {len(queued)} Job Pod(s) are queued; inspect with 'queue' or override with --allow-queued")
    response = client.submit_yaml(namespace=metadata["namespace"], yaml_text=yaml_text)
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(f"Submitted Job {metadata['namespace']}/{metadata['name']}")
    return 0


def command_jobs(args: argparse.Namespace) -> int:
    response = _build_client(args).list_jobs(namespace=args.namespace, keyword=args.keyword, page_size=args.page_size)
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    rows = [_job_row(item) for item in _items(response)]
    print_table(
        rows,
        [
            ("name", "NAME"),
            ("status", "STATUS"),
            ("active", "ACTIVE"),
            ("succeeded", "SUCCEEDED"),
            ("failed", "FAILED"),
            ("parallelism", "PARALLEL"),
            ("created", "CREATED"),
        ],
    )
    return 0


def _filtered_pod_items(response: Mapping[str, Any], job_name: Optional[str]) -> List[Dict[str, Any]]:
    items = _items(response)
    return [item for item in items if _job_name_for_pod(item) == job_name] if job_name else items


def _list_all_pods(client: BingoClient, *, namespace: str, keyword: str, page_size: int) -> List[Dict[str, Any]]:
    all_items: List[Dict[str, Any]] = []
    page = 1
    while True:
        response = client.list_pods(namespace=namespace, keyword=keyword, page=page, page_size=page_size)
        page_items = _items(response)
        all_items.extend(page_items)
        data = _response_data(response)
        count = data.get("count") if isinstance(data, Mapping) else None
        if not page_items or len(page_items) < page_size or (isinstance(count, int) and len(all_items) >= count):
            return all_items
        page += 1


def _is_finished_pod(item: Mapping[str, Any]) -> bool:
    phase = _status_mapping(item).get("phase") or item.get("phase")
    if phase:
        return str(phase).lower() in {"completed", "succeeded", "failed", "error"}
    return _pod_status(item).lower() in {"completed", "succeeded", "failed", "error"}


def _finished_pods(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((item for item in items if _is_finished_pod(item)), key=lambda item: (_created_at(item), str(_metadata(item).get("name") or "")))


def command_pods(args: argparse.Namespace) -> int:
    response = _build_client(args).list_pods(
        namespace=args.namespace,
        keyword=args.job or args.keyword,
        page_size=args.page_size,
    )
    items = _filtered_pod_items(response, args.job)
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return 0
    print_table(
        [_pod_row(item) for item in items],
        [("name", "NAME"), ("status", "STATUS"), ("job", "JOB"), ("node", "NODE"), ("created", "CREATED")],
    )
    return 0


def command_queue(args: argparse.Namespace) -> int:
    queued = _queued_pods(
        _list_all_pods(
            _build_client(args),
            namespace=args.namespace,
            keyword=args.keyword,
            page_size=args.page_size,
        )
    )
    if args.json:
        print(json.dumps(queued, ensure_ascii=False, indent=2))
    else:
        print_table(
            [_queue_row(item) for item in queued],
            [("pod", "POD"), ("job", "JOB"), ("reason", "REASON"), ("message", "MESSAGE"), ("created", "CREATED")],
        )
        if queued:
            print(f"{len(queued)} queued Job Pod(s) found.")
        else:
            print("No queued Job Pods.")
    return 1 if queued and args.fail_if_queued else 0


def command_cleanup_pods(args: argparse.Namespace) -> int:
    client = _build_client(args)
    candidates = _finished_pods(
        _list_all_pods(
            client,
            namespace=args.namespace,
            keyword=args.keyword,
            page_size=args.page_size,
        )
    )
    print_table(
        [_pod_row(item) for item in candidates],
        [("name", "NAME"), ("status", "STATUS"), ("job", "JOB"), ("node", "NODE"), ("created", "CREATED")],
    )
    if not candidates:
        print("No finished Pods to delete.")
        return 0
    if args.dry_run:
        print(f"Dry run only; would delete {len(candidates)} finished Pods.")
        return 0

    targets: List[Tuple[str, str]] = []
    for item in candidates:
        metadata = _metadata(item)
        namespace = str(metadata.get("namespace") or args.namespace)
        name = str(metadata.get("name") or item.get("Name") or "")
        if not name:
            raise PlatformError("refusing to delete a finished Pod whose name is missing")
        _validate_dns_label(namespace, "Pod namespace")
        _validate_dns_label(name, "Pod name")
        targets.append((namespace, name))

    if not args.yes:
        expected = f"DELETE {len(candidates)}"
        try:
            confirmation = input(f"Type {expected!r} to delete the listed Pods: ")
        except EOFError:
            confirmation = ""
        if confirmation != expected:
            print("Aborted; no Pods were deleted.")
            return 1

    failures: List[Tuple[str, PlatformError]] = []
    for namespace, name in targets:
        try:
            client.delete_pod(namespace=namespace, name=name)
            print(f"Deleted Pod {namespace}/{name}")
        except PlatformError as exc:
            failures.append((name, exc))
            print(f"Failed to delete Pod {namespace}/{name}: {exc}", file=sys.stderr)
    print(f"Deleted {len(candidates) - len(failures)}/{len(candidates)} finished Pods.")
    return 1 if failures else 0


def _status_once(client: BingoClient, args: argparse.Namespace) -> None:
    job_response = client.job_detail(namespace=args.namespace, name=args.name)
    pods_response = client.list_pods(namespace=args.namespace, keyword=args.name, page_size=args.page_size)
    pods = _filtered_pod_items(pods_response, args.name)
    if args.json:
        print(json.dumps({"job": _response_data(job_response), "pods": pods}, ensure_ascii=False, indent=2))
        return
    print(f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] Job")
    print_table(
        [_job_row(_response_data(job_response))],
        [
            ("name", "NAME"),
            ("status", "STATUS"),
            ("active", "ACTIVE"),
            ("succeeded", "SUCCEEDED"),
            ("failed", "FAILED"),
            ("created", "CREATED"),
        ],
    )
    print("\nPods")
    print_table(
        [_pod_row(item) for item in pods],
        [("name", "NAME"), ("status", "STATUS"), ("node", "NODE"), ("created", "CREATED")],
    )


def command_status(args: argparse.Namespace) -> int:
    client = _build_client(args)
    while True:
        _status_once(client, args)
        if not args.watch:
            return 0
        time.sleep(args.interval)
        print()


def command_events(args: argparse.Namespace) -> int:
    client = _build_client(args)
    creation_timestamp = args.creation_timestamp
    if not creation_timestamp and args.kind == "Job":
        detail = client.job_detail(namespace=args.namespace, name=args.name)
        creation_timestamp = _metadata(_response_data(detail)).get("creationTimestamp")
    if not creation_timestamp:
        raise PlatformError("--creation-timestamp is required when the resource timestamp cannot be discovered")
    response = client.list_events(
        namespace=args.namespace,
        name=args.name,
        creation_timestamp=str(creation_timestamp),
        kind=args.kind,
        page_size=args.page_size,
    )
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    print_table(
        [_event_row(item) for item in _items(response)],
        [
            ("last", "LAST"),
            ("type", "TYPE"),
            ("reason", "REASON"),
            ("object", "OBJECT"),
            ("component", "COMPONENT"),
            ("message", "MESSAGE"),
        ],
    )
    return 0


def _add_render_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("template", type=Path, help="Job YAML template")
    parser.add_argument("--name", help="override metadata.name and the pod job-name label")
    parser.add_argument("--namespace", help="override metadata.namespace")
    parser.add_argument("--container", help="container name to modify; defaults to the first container")
    parser.add_argument("--shell-command", help="replace command/args with: bash -lc COMMAND")
    parser.add_argument("--command-json", help='replace container command with a JSON string array, e.g. \'["bash","script.sh"]\'')
    parser.add_argument("--args-json", help="replace container args with a JSON string array")
    parser.add_argument("--cpu", help="set identical CPU request and limit")
    parser.add_argument("--memory", help="set identical memory request and limit, e.g. 300Gi")
    parser.add_argument("--gpus", help="set identical nvidia.com/vgpu request and limit")
    parser.add_argument("--vgpu-cores", help="set identical nvidia.com/vgpu-cores request and limit")
    parser.add_argument("--vgpu-mem", "--vgpu-memory", dest="vgpu_mem", help="set identical nvidia.com/vgpu-mem request and limit")
    parser.add_argument("-o", "--output", type=Path, help="write the rendered YAML to this path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_namespace = os.environ.get("BINGO_NAMESPACE", DEFAULT_NAMESPACE)
    parser.add_argument("--base-url", default=os.environ.get("BINGO_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--dex-url", default=os.environ.get("BINGO_DEX_URL", DEFAULT_DEX_URL))
    parser.add_argument("--cluster-id", default=os.environ.get("BINGO_CLUSTER_ID", DEFAULT_CLUSTER_ID))
    parser.add_argument("--workspace-id", default=os.environ.get("BINGO_WORKSPACE_ID", DEFAULT_WORKSPACE_ID))
    parser.add_argument("--username", default=os.environ.get("BINGO_USERNAME", DEFAULT_USERNAME))
    parser.add_argument("--proxy-url", default=os.environ.get("BINGO_PROXY", DEFAULT_PROXY_URL), help="HTTP CONNECT proxy used for the private platform")
    parser.add_argument("--no-proxy", action="store_true", help="connect to the private platform directly")
    parser.add_argument("--verify-tls", action="store_true", help="verify the platform TLS certificate (disabled by default for the private IP certificate)")
    parser.add_argument("--no-token-cache", action="store_true", help="do not reuse the short-lived access token from ~/.cache/xmetai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="render a patched Job YAML without contacting the platform")
    _add_render_options(render_parser)
    render_parser.set_defaults(handler=command_render)

    submit_parser = subparsers.add_parser("submit", help="render and submit a Job YAML")
    _add_render_options(submit_parser)
    submit_parser.add_argument("--dry-run", action="store_true", help="render only; do not contact the platform")
    submit_parser.add_argument("--allow-queued", action="store_true", help="submit even when another Job Pod is waiting for scheduling")
    submit_parser.add_argument("--json", action="store_true", help="print the raw API response")
    submit_parser.set_defaults(handler=command_submit)

    jobs_parser = subparsers.add_parser("jobs", help="list Jobs")
    jobs_parser.add_argument("--namespace", default=default_namespace)
    jobs_parser.add_argument("--keyword", default="")
    jobs_parser.add_argument("--page-size", type=int, default=50)
    jobs_parser.add_argument("--json", action="store_true")
    jobs_parser.set_defaults(handler=command_jobs)

    pods_parser = subparsers.add_parser("pods", help="list Pods")
    pods_parser.add_argument("--namespace", default=default_namespace)
    pods_parser.add_argument("--keyword", default="")
    pods_parser.add_argument("--job", help="show only Pods owned by this Job")
    pods_parser.add_argument("--page-size", type=int, default=50)
    pods_parser.add_argument("--json", action="store_true")
    pods_parser.set_defaults(handler=command_pods)

    queue_parser = subparsers.add_parser("queue", aliases=["queued"], help="list Job Pods that are waiting to be scheduled")
    queue_parser.add_argument("--namespace", default=default_namespace)
    queue_parser.add_argument("--keyword", default="", help="limit results by Pod name")
    queue_parser.add_argument("--page-size", type=int, default=100)
    queue_parser.add_argument("--json", action="store_true")
    queue_parser.add_argument("--fail-if-queued", action="store_true", help="exit with status 1 when queued Job Pods exist")
    queue_parser.set_defaults(handler=command_queue)

    cleanup_parser = subparsers.add_parser(
        "cleanup-pods",
        aliases=["cleanup-finished-pods", "cleanup-completed-pods"],
        help="list and delete all succeeded or failed Pods after confirmation",
    )
    cleanup_parser.add_argument("--namespace", default=default_namespace)
    cleanup_parser.add_argument("--keyword", default="", help="limit candidates by Pod name")
    cleanup_parser.add_argument("--page-size", type=int, default=100)
    cleanup_parser.add_argument("--dry-run", action="store_true", help="list candidates without prompting or deleting")
    cleanup_parser.add_argument("--yes", action="store_true", help="delete the listed candidates without an interactive prompt")
    cleanup_parser.set_defaults(handler=command_cleanup_pods)

    status_parser = subparsers.add_parser("status", help="show one Job and its Pods")
    status_parser.add_argument("name")
    status_parser.add_argument("--namespace", default=default_namespace)
    status_parser.add_argument("--page-size", type=int, default=50)
    status_parser.add_argument("--watch", action="store_true", help="refresh until interrupted")
    status_parser.add_argument("--interval", type=float, default=10.0)
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(handler=command_status)

    events_parser = subparsers.add_parser("events", help="show Kubernetes events for a resource")
    events_parser.add_argument("name")
    events_parser.add_argument("--kind", default="Job")
    events_parser.add_argument("--namespace", default=default_namespace)
    events_parser.add_argument("--creation-timestamp", help="resource creation time; discovered automatically for Jobs")
    events_parser.add_argument("--page-size", type=int, default=50)
    events_parser.add_argument("--json", action="store_true")
    events_parser.set_defaults(handler=command_events)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "page_size", 1) < 1:
        parser.error("--page-size must be positive")
    if getattr(args, "interval", 1) <= 0:
        parser.error("--interval must be positive")
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except PlatformError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
