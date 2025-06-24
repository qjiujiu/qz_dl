import os
import time
import logging
import colorlog
from types import MethodType
from typing import TypeVar

try:
    from typing import Protocol 
except ImportError:
    from typing_extensions import Protocol 
    

# 定义泛型 Logger 类型
LogX = TypeVar("LogX", bound=logging.Logger)


# 自定义的 LoggerX 协议
class LoggerX(Protocol):
    """自定义 Logger 协议，扩展 logging.Logger 增加 success 和 highlight，并包含所有原生方法"""

    # 原生 logging 方法
    def debug(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def info(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def warning(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def error(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def critical(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def exception(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def log(self: LogX, level: int, msg: str, *args, **kwargs) -> None: ...

    # 扩展的 success（等价于 info）和 highlight（等价于 debug）
    def success(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def highlight(self: LogX, msg: str, *args, **kwargs) -> None: ...
    
    # 是否开启 debug 模块的开关
    def is_debug(self, enable=True):...


# 日志模块 log_level 控制记录何种级别的日志，低于这个级别的日志不被记录 debug < info < warning < error < critical
def logger_initiate(log_level=logging.INFO, is_console=True, is_file=True, is_colorful=False) -> LoggerX:
    """日志基础设置，支持本地控制台和彩色日志文件输出"""
    # 统一日志格式
    log_format = "[%(asctime)s] %(levelname)s - %(pathname)s[line:%(lineno)d]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 普通日志格式
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 彩色日志格式
    colorful_formatter = colorlog.ColoredFormatter(
        "%(log_color)s" + log_format,  # 保持结构一致，仅增加颜色
        datefmt=date_format,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # 配置 logger
    logging.basicConfig(format=log_format, level=log_level)
    _logger = logging.getLogger("autotest")

    # 每次调用这个函数之前需要清除已有的 handler，以免日志重复写入
    if _logger.hasHandlers():
        _logger.handlers.clear()

    _logger.setLevel(log_level)

    # 输出到控制台
    if is_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colorful_formatter if is_colorful else formatter)
        console_handler.setLevel(log_level)
        _logger.addHandler(console_handler)

    # 输出到文件，文件写入当前正在运行的目录
    if is_file:
        current_time = time.strftime("%Y%m%d")
        log_filename = f"{current_time}.log"

        relative_log_directory = "logs"
        current_path = os.getcwd()
        log_directory = os.path.join(current_path, relative_log_directory)

        if not os.path.exists(log_directory):
            os.makedirs(log_directory, exist_ok=True)

        file_handler = logging.FileHandler(f"{log_directory}/{log_filename}")
        file_handler.setLevel(log_level)

        # 文件日志使用普通格式
        file_handler.setFormatter(formatter)

        # 每个 handler 处理一个日志，写入文件的这个handler也会打印于终端，需要手动禁用日志传播
        _logger.addHandler(file_handler)
        _logger.propagate = False

    # 动态添加 highlight（等价于 debug）与 success（等价于 info）
    setattr(_logger, "highlight", _logger.debug)
    setattr(_logger, "success", _logger.info)

    return _logger


# 初始化 logger
logger = logger_initiate(log_level=logging.INFO, is_console=True, is_file=True, is_colorful=True)


# 动态绑定 is_debug 方法
def is_debug(self, enable=True):
    """
    动态修改日志级别为 DEBUG 或恢复为 INFO
    :param enable: 是否启用 DEBUG 级别
    """
    if enable:
        new_level = logging.DEBUG
    else:
        new_level = logging.INFO

    # 修改全局日志级别
    self.setLevel(new_level)

    # 修改每个 handler 的日志级别
    for handler in self.handlers:
        handler.setLevel(new_level)

# 将 is_debug 方法绑定到 logger 对象
logger.is_debug = MethodType(is_debug, logger)


# 允许被外部导入
__all__ = ["logger_initiate", "logging", "logger"]


# 测试代码
if __name__ == '__main__':
    # 默认日志级别为 INFO
    logger.info("这是 INFO 消息")
    logger.debug("这是 DEBUG 消息（默认不显示）")

    # 启用 DEBUG 级别
    logger.is_debug(True)
    logger.info("DEBUG 级别已启用...")
    logger.debug("这是 DEBUG 消息,现在应该显示!")

    # 恢复为 INFO 级别
    logger.is_debug(False)
    logger.info("恢复 INFO 级别...")
    logger.debug("这是 DEBUG 消息，恢复之后不显示！")