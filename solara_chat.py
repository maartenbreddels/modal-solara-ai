import solara
import solara.lab

import modal
from typing import List
from typing_extensions import TypedDict

system_prompt = (
    "be concise!. You are in the role of a human named Lee Bruce who is on a phone."
)


class MessageDict(TypedDict):
    role: str  # "user", "assistant" or "system"
    content: str


messages: solara.Reactive[List[MessageDict]] = solara.reactive(
    [{"role": "system", "content": system_prompt}]
)


@solara.lab.task
def promt_ai(model, message: str):
    messages.value = [
        *messages.value,
        {"role": "user", "content": message},
    ]
    # start with an empty reply message, so we render and empty message in the chat
    # while the AI is thinking, and when the token stream in, we append it to the last message
    messages.value = [*messages.value, {"role": "assistant", "content": ""}]
    history = messages.value.copy()

    # call the model running at Modal
    for delta in model.predict.remote_gen(message, history):
        if not promt_ai.is_current():
            return  # we got cancelled or replace, abort
        # replace the last message element with the appended content
        updated_message: MessageDict = {
            "role": "assistant",
            "content": messages.value[-1]["content"] + delta,
        }
        # replace the last message element with the appended content
        # which will update the UI
        messages.value = [*messages.value[:-1], updated_message]


def make_model():
    ModelCls = modal.Cls.lookup("solara-llm-demo", "Model")
    model = ModelCls()
    return model


@solara.component
def Page():
    # create the model once
    model = solara.use_memo(make_model, [])
    # print("promt_ai", promt_ai.result)

    if model is None:
        solara.Error(
            "Could not connect to the model running at Modal, please check if the llm Modal app is deployed."
        )
        return

    with solara.Column(
        style={
            "width": "100%",
            "height": "100%",
            "padding": "24px",
            "padding-bottom": "64px",
        },
    ):
        with solara.lab.ChatBox():
            for item in messages.value:
                if item["role"] == "system":
                    continue  # skip the system message
                with solara.lab.ChatMessage(
                    user=item["role"] == "user",
                    avatar=False,
                    name="MyAssistant" if item["role"] == "assistant" else "You",
                    color=(
                        "rgba(0,0,0, 0.06)"
                        if item["role"] == "assistant"
                        else "#ff991f"
                    ),
                    avatar_background_color=(
                        "primary" if item["role"] == "assistant" else None
                    ),
                    border_radius="20px",
                ):
                    solara.Markdown(item["content"])
        if promt_ai.pending:
            solara.Text(
                "I'm thinking...", style={"font-size": "1rem", "padding-left": "20px"}
            )
            solara.ProgressLinear()
        solara.lab.ChatInput(
            send_callback=lambda msg: promt_ai(model, msg),
            disabled=promt_ai.pending,
            autofocus=True,
        )
