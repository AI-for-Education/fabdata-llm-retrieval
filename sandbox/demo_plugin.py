from pathlib import Path
import asyncio
import os
import json

from dotenv import load_dotenv

load_dotenv(override=True)

from fdllm import get_caller
from fdllm.sysutils import register_models
from fdllm.chat import ChatController
from fdllmret.plugin import retrieval_plugin
from fdllmret.helpers.encoding import DocsetEncoding

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

register_models(Path.home() / ".fdllm/custom_models.yaml")


async def create_chatcontroller(caller="claude-3-5-sonnet-20241022"):
    with open(HERE / "contexts/context_searcher_3.txt") as f:
        msg = "\n".join(ln for ln in f if ln.strip() and ln.strip()[0] != "#")
    caller = get_caller(caller)
    if caller.Model.Flexible_SysMsg:
        controller = ChatController(Caller=caller, Sys_Msg={0: msg, -1: msg})
    else:
        controller = ChatController(Caller=caller, Sys_Msg={0: msg})
    plugin, datastore = await retrieval_plugin(
        dbhost=os.environ.get("REDIS_HOST"),
        dbport=os.environ.get("REDIS_PORT"),
        dbssl=os.environ.get("REDIS_SSL"),
        dbauth=os.environ.get("REDIS_PASSWORD"),
    )
    controller.register_plugin(plugin)
    return controller, datastore


async def main():
    # model = "claude-3-opus-20240229"
    # model = "claude-3-5-sonnet-20241022"
    # model = "claude-3-5-haiku-20241022"
    # model = "gpt-4o-2024-08-06"
    model = "gpt-4o-mini-2024-07-18"
    controller, datastore = await create_chatcontroller(caller=model)
    print_tool_results = True
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "exit":
            break
        _, output = await controller.achat(prompt, max_tokens=1000, temperature=1)
        for msg in controller.recent_history[1:]:
            if msg.Message:
                print(msg.Message)
                print()
            if print_tool_results and msg.ToolCalls is not None:
                for tc in msg.ToolCalls:
                    if tc.Response is not None:
                        print(json.dumps(tc.Args, indent=4))
                        print(json.dumps(json.loads(tc.Response), indent=4))
        print("#" * 50)
        # print(controller.recent_tool_calls)
        # print(output.Message)
    await datastore.client.connection_pool.disconnect()


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
