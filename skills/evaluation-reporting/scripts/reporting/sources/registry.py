"""
组件注册机制

按照 README 第 13.5 节实现显式注册，支持 reader/transform/metric/step/visualization。
注册名称必须唯一，未注册组件在 validate 阶段失败。
"""

from typing import Dict, Callable, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from reporting.sources.errors import ConfigError


class ComponentType(Enum):
    """组件类型"""

    READER = "reader"
    TRANSFORM = "transform"
    METRIC = "metric"
    STEP = "step"
    VISUALIZATION = "visualization"
    WRITER = "writer"
    PROTOCOL = "protocol"


@dataclass
class ComponentDescriptor:
    """组件描述符"""

    name: str  # 注册名称
    component_type: ComponentType
    version: str
    factory: Callable  # 工厂函数，返回组件实例
    description: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None  # 能力描述（如支持的维度、产品类型）
    dependencies: Optional[List[str]] = None  # 依赖提示（如 cfgrib、scipy）


class Registry:
    """
    全局组件注册表

    按照 README 第 13.5 节：
    - 注册时发现重复名称必须失败
    - 未注册名称必须在 validate 阶段失败
    - 配置只能引用注册名和受校验的参数
    """

    def __init__(self):
        self._components: Dict[str, ComponentDescriptor] = {}

    def register(
        self,
        name: str,
        component_type: ComponentType,
        version: str,
        factory: Callable,
        description: Optional[str] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        注册组件

        Args:
            name: 组件名称，格式 "type.name"（如 "reader.netcdf"）
            component_type: 组件类型
            version: 版本号
            factory: 工厂函数
            description: 描述
            capabilities: 能力描述
            dependencies: 依赖提示

        Raises:
            ConfigError: 如果名称已存在
        """
        full_name = f"{component_type.value}.{name}"

        if full_name in self._components:
            existing = self._components[full_name]
            raise ConfigError(
                f"Component '{full_name}' already registered (version {existing.version}). "
                f"Cannot re-register with version {version}."
            )

        descriptor = ComponentDescriptor(
            name=full_name,
            component_type=component_type,
            version=version,
            factory=factory,
            description=description,
            capabilities=capabilities,
            dependencies=dependencies,
        )

        self._components[full_name] = descriptor

    def get(self, name: str, component_type: Optional[ComponentType] = None) -> ComponentDescriptor:
        """
        获取组件描述符

        Args:
            name: 组件名称，可以是 "type.name" 或 "name"
            component_type: 如果 name 不包含类型前缀，必须提供此参数

        Returns:
            ComponentDescriptor

        Raises:
            ConfigError: 如果组件未注册
        """
        # 如果 name 不包含 "."，需要补充类型前缀
        if "." not in name:
            if component_type is None:
                raise ConfigError(
                    f"Component name '{name}' does not include type prefix "
                    f"and component_type not provided"
                )
            full_name = f"{component_type.value}.{name}"
        else:
            full_name = name

        if full_name not in self._components:
            raise ConfigError(
                f"Component '{full_name}' not registered. "
                f"Available: {', '.join(sorted(self._components.keys()))}"
            )

        return self._components[full_name]

    def list_components(
        self, component_type: Optional[ComponentType] = None
    ) -> List[ComponentDescriptor]:
        """
        列出已注册组件

        Args:
            component_type: 可选，只返回指定类型的组件

        Returns:
            组件描述符列表
        """
        if component_type is None:
            return list(self._components.values())

        prefix = f"{component_type.value}."
        return [desc for name, desc in self._components.items() if name.startswith(prefix)]

    def names(self, component_type: Optional[ComponentType] = None) -> List[str]:
        """
        列出已注册组件的短名称（不带类型前缀）

        Args:
            component_type: 可选，只返回指定类型

        Returns:
            按字典序排列的短名称列表
        """
        return sorted(desc.name.split(".", 1)[1] for desc in self.list_components(component_type))

    def build(self, component_type: ComponentType, name: str, **params: Any) -> Any:
        """
        按类型和名称构造组件实例

        这是配置解析与组件实现之间的唯一入口：调用方只提供注册名和参数，
        不允许根据名称分支判断该构造哪个类。

        Args:
            component_type: 组件类型
            name: 注册名称（可带或不带类型前缀）
            **params: 传给工厂函数的参数

        Returns:
            工厂函数返回的组件实例

        Raises:
            ConfigError: 组件未注册或构造失败
        """
        descriptor = self.get(name, component_type)
        try:
            return descriptor.factory(**params)
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(
                f"Failed to build component '{descriptor.name}': {exc}"
            ) from exc

    def validate_reference(self, name: str, component_type: ComponentType) -> None:
        """
        验证配置引用是否有效

        Args:
            name: 组件名称
            component_type: 期望的组件类型

        Raises:
            ConfigError: 如果引用无效
        """
        try:
            descriptor = self.get(name, component_type)
            if descriptor.component_type != component_type:
                raise ConfigError(
                    f"Component '{name}' is registered as {descriptor.component_type.value}, "
                    f"not {component_type.value}"
                )
        except ConfigError:
            raise  # 重新抛出，保持错误消息


# 全局单例注册表
_global_registry = Registry()


def get_registry() -> Registry:
    """获取全局注册表"""
    return _global_registry


def register_reader(
    name: str,
    version: str,
    factory: Callable,
    description: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    dependencies: Optional[List[str]] = None,
) -> None:
    """便捷函数：注册 Reader"""
    _global_registry.register(
        name=name,
        component_type=ComponentType.READER,
        version=version,
        factory=factory,
        description=description,
        capabilities=capabilities,
        dependencies=dependencies,
    )


def register_metric(
    name: str,
    version: str,
    factory: Callable,
    description: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    dependencies: Optional[List[str]] = None,
) -> None:
    """便捷函数：注册 Metric"""
    _global_registry.register(
        name=name,
        component_type=ComponentType.METRIC,
        version=version,
        factory=factory,
        description=description,
        capabilities=capabilities,
        dependencies=dependencies,
    )


def register_transform(
    name: str,
    version: str,
    factory: Callable,
    description: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    dependencies: Optional[List[str]] = None,
) -> None:
    """便捷函数：注册 Transform"""
    _global_registry.register(
        name=name,
        component_type=ComponentType.TRANSFORM,
        version=version,
        factory=factory,
        description=description,
        capabilities=capabilities,
        dependencies=dependencies,
    )


def register_writer(
    name: str,
    version: str,
    factory: Callable,
    description: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    dependencies: Optional[List[str]] = None,
) -> None:
    """便捷函数：注册结果输出视图"""
    _global_registry.register(
        name=name,
        component_type=ComponentType.WRITER,
        version=version,
        factory=factory,
        description=description,
        capabilities=capabilities,
        dependencies=dependencies,
    )


def register_protocol(
    name: str,
    version: str,
    factory: Callable,
    description: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    dependencies: Optional[List[str]] = None,
) -> None:
    """便捷函数：注册验证协议"""
    _global_registry.register(
        name=name,
        component_type=ComponentType.PROTOCOL,
        version=version,
        factory=factory,
        description=description,
        capabilities=capabilities,
        dependencies=dependencies,
    )
