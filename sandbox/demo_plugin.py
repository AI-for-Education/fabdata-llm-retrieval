from pathlib import Path
import asyncio
import os

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


async def create_chatcontroller(caller="gpt-4-1106-preview"):
    with open(HERE / "contexts/context_searcher.txt") as f:
        msg = "\n".join(ln for ln in f if ln.strip() and ln.strip()[0] != "#")
    controller = ChatController(Caller=get_caller(caller), Sys_Msg={0: msg, -1: msg})
    plugin, datastore = await retrieval_plugin(
        dbhost=os.environ.get("REDIS_HOST"),
        dbport=os.environ.get("REDIS_PORT"),
        dbssl=os.environ.get("REDIS_SSL"),
        dbauth=os.environ.get("REDIS_PASSWORD"),
    )
    controller.register_plugin(plugin)
    return controller, datastore


async def main():
    controller, datastore = await create_chatcontroller(caller="gpt-4-1106-preview")
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "exit":
            break
        _, output = await controller.achat(prompt, max_tokens=1000)
        print(controller.recent_tool_calls)
        print(output.Message)
    await datastore.client.connection_pool.disconnect()


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
