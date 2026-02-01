from typing import Callable, Any
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts


# 工具调用中间件 工具执行监控
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,  # 模型请求数据封装
        handler: Callable[[ToolCallRequest], ToolMessage | Command]  # 执行函数本身
) -> ToolMessage | Command:
    logger.info(f"[tool monitor]执行工具: {request.tool_call['name']}")
    logger.info(f"[tool monitor]参数: {request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == 'fill_context_for_report':  # 填充报告生成场景
            logger.info(f"[tool monitor]fill_context_for_report工具被调用，注入上下文 report=True")
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.info(f"工具{request.tool_call['name']}调用失败: {e}")
        raise


# 模型执行前输出日志
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    logger.info(f"[log_before_model]: 即将调用模型，带有{len(state['messages'])}条消息，消息如下：")
    # for message in state['messages']:
    #     logger.info(f"[log_before_model][{type(message).__name__}]: {message.content.strip()}")
    logger.info(f"[log_before_model]: ----------省略已输出内容----------")
    logger.info(f"[log_before_model][{type(state['messages'][-1]).__name__}]: {state['messages'][-1].content.strip()}")

    return None


# 动态切换提示词
@dynamic_prompt  # 每一次生成提示词之前都会调用
def report_prompt_switch(request: ModelRequest) -> str:
    is_report = request.runtime.context.get("report", False)
    if is_report:  # 返回报告生成场景的提示词
        return load_report_prompts()

    return load_system_prompts()  # 返回正常场景的提示词
