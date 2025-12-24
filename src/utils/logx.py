import os
import logging
import colorlog

from typing import TypeVar, Optional, cast
from typing_extensions import Protocol
from functools import partial


LogX = TypeVar("LogX", bound=logging.Logger)

is_debug_mode = True


# 注册级别名称，这样日志里就会显示 [HIGHLIGHT] 或 [SUCCESS] 而不是 [Level 15]
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# 高于 DEBUG，低于 INFO
LEVEL_HIGHLIGHT = 15
logging.addLevelName(LEVEL_HIGHLIGHT, "HIGHLIGHT")

# 高于 INFO ，低于 WARNING
LEVEL_SUCCESS = 25
logging.addLevelName(LEVEL_SUCCESS, "SUCCESS")




class LoggerX(Protocol):
    """自定义 Logger 协议，用于代码提示"""

    def debug(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def info(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def warning(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def error(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def critical(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def exception(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def log(self: LogX, level: int, msg: str, *args, **kwargs) -> None: ...

    # 扩展方法
    def success(self: LogX, msg: str, *args, **kwargs) -> None: ...
    def highlight(self: LogX, msg: str, *args, **kwargs) -> None: ...


def get_extended_logger(name: Optional[str] = None) -> LoggerX:
    """获取增强版 Logger。

    逻辑：
    1. 获取现有的 logger 实例（复用 bytedlogger 或 basicConfig 的配置）。
    2. 动态注入 success 和 highlight 方法。
    3. 如果环境变量 DEBUG=1，则将控制台输出改为彩色格式。
    """

    _logger = logging.getLogger(name)

    # 使用 使用自定义级别, 注入扩展方法
    # 使用 partial 绑定 log 方法和具体的 level，实现类似于 debug()/info() 的调用方式

    if not hasattr(_logger, "success"):
        # 相当于 _logger.log(25, msg, *args, **kwargs)
        setattr(_logger, "success", partial(_logger.log, LEVEL_SUCCESS))

    if not hasattr(_logger, "highlight"):
        setattr(_logger, "highlight", partial(_logger.log, LEVEL_HIGHLIGHT))


    if is_debug_mode:
        _enable_color_formatting(_logger)
        _logger.setLevel(logging.DEBUG)

    return cast(LoggerX, _logger)


def _enable_color_formatting(logger: logging.Logger):
    """
    仅在 DEBUG 模式下调用
    """
    log_format = "[%(asctime)s] %(levelname)s - %(pathname)s[line:%(lineno)d]: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 配置颜色映射
    colorful_formatter = colorlog.ColoredFormatter(
        "%(log_color)s" + log_format,
        datefmt=date_format,
        log_colors={
            'DEBUG': '',
            'HIGHLIGHT': 'cyan',
            'INFO': 'white',
            'SUCCESS': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        },
        # secondary_log_colors 用于更细粒度的控制，这里不需要
    )

    target_logger = logger
    if not target_logger.handlers and target_logger.parent:
        target_logger = target_logger.parent

    found_console = False
    if target_logger:
        for handler in target_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(colorful_formatter)
                found_console = True

    if not found_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colorful_formatter)
        logger.addHandler(console_handler)


# 默认导出的实例，复用 root logger 或指定名称
# 使用 logging 这个名字, 以便到时候直接替换默认的日志库
logger = logx = get_extended_logger()

__all__ = ["logx", "logger", "logging", "get_extended_logger"]


if __name__ == '__main__':
    # 模拟设置环境变量 (你在本地运行时手动 export DEBUG=1)
    # os.environ["DEBUG"] = "1"

    # 重新获取 logx，此时它应该检测到了环境变量
    test_logger = get_extended_logger()

    print("--- 测试日志输出 ---")
    test_logger.debug("这是 DEBUG 消息 (Cyan)")
    test_logger.highlight("这是 HIGHLIGHT 消息 (Cyan)")
    test_logger.info("这是 INFO 消息 (Green)")
    test_logger.success("这是 SUCCESS 消息 (Green)")
    test_logger.warning("这是 WARNING 消息 (Yellow)")
    test_logger.error("这是 ERROR 消息 (Red)")
