"""Conversation history."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import chat_session, intent, llm, template
from homeassistant.util.hass_dict import HassKey
from homeassistant.util.json import JsonObjectType

from . import trace
from .const import DOMAIN
from .models import ConversationInput, ConversationResult

DATA_CHAT_HISTORY: HassKey[dict[str, ChatLog]] = HassKey("conversation_chat_log")

LOGGER = logging.getLogger(__name__)


@contextmanager
def async_get_chat_log(
    hass: HomeAssistant,
    session: chat_session.ChatSession,
    user_input: ConversationInput | None = None,
) -> Generator[ChatLog]:
    """Return chat log for a specific chat session."""
    all_history = hass.data.get(DATA_CHAT_HISTORY)
    if all_history is None:
        all_history = {}
        hass.data[DATA_CHAT_HISTORY] = all_history

    history = all_history.get(session.conversation_id)

    if history:
        history = replace(history, content=history.content.copy())
    else:
        history = ChatLog(hass, session.conversation_id)

        @callback
        def do_cleanup() -> None:
            """Handle cleanup."""
            all_history.pop(session.conversation_id)

        session.async_on_cleanup(do_cleanup)

    if user_input is not None:
        history.async_add_user_content(UserContent(content=user_input.text))

    last_message = history.content[-1]

    yield history

    if history.content[-1] is last_message:
        LOGGER.debug(
            "History opened but no assistant message was added, ignoring update"
        )
        return

    all_history[session.conversation_id] = history


class ConverseError(HomeAssistantError):
    """Error during initialization of conversation.

    Will not be stored in the history.
    """

    def __init__(
        self, message: str, conversation_id: str, response: intent.IntentResponse
    ) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.conversation_id = conversation_id
        self.response = response

    def as_conversation_result(self) -> ConversationResult:
        """Return the error as a conversation result."""
        return ConversationResult(
            response=self.response,
            conversation_id=self.conversation_id,
        )


@dataclass(frozen=True)
class SystemContent:
    """Base class for chat messages."""

    role: str = field(init=False, default="system")
    content: str


@dataclass(frozen=True)
class UserContent:
    """Assistant content."""

    role: str = field(init=False, default="user")
    content: str


@dataclass(frozen=True)
class AssistantContent:
    """Assistant content."""

    role: str = field(init=False, default="assistant")
    agent_id: str
    content: str
    tool_calls: list[llm.ToolInput] | None = None


@dataclass(frozen=True)
class ToolResultContent:
    """Tool result content."""

    role: str = field(init=False, default="tool_result")
    agent_id: str
    tool_call_id: str
    tool_name: str
    tool_result: JsonObjectType


Content = SystemContent | UserContent | AssistantContent | ToolResultContent


@dataclass
class ChatLog:
    """Class holding the chat history of a specific conversation."""

    hass: HomeAssistant
    conversation_id: str
    content: list[Content] = field(default_factory=lambda: [SystemContent(content="")])
    extra_system_prompt: str | None = None
    llm_api: llm.APIInstance | None = None

    @callback
    def async_add_user_content(self, content: UserContent) -> None:
        """Add user content to the log."""
        self.content.append(content)

    async def async_add_assistant_content(
        self, content: AssistantContent
    ) -> AsyncGenerator[ToolResultContent]:
        """Add assistant content."""
        self.content.append(content)

        if content.tool_calls is None:
            return

        if self.llm_api is None:
            raise ValueError("No LLM API configured")

        for tool_input in content.tool_calls:
            LOGGER.debug(
                "Tool call: %s(%s)", tool_input.tool_name, tool_input.tool_args
            )

            try:
                tool_result = await self.llm_api.async_call_tool(tool_input)
            except (HomeAssistantError, vol.Invalid) as e:
                tool_result = {"error": type(e).__name__}
                if str(e):
                    tool_result["error_text"] = str(e)
            LOGGER.debug("Tool response: %s", tool_result)

            response_content = ToolResultContent(
                agent_id=content.agent_id,
                tool_call_id=tool_input.id,
                tool_name=tool_input.tool_name,
                tool_result=tool_result,
            )
            self.content.append(response_content)
            yield response_content

    async def async_update_llm_data(
        self,
        conversing_domain: str,
        user_input: ConversationInput,
        user_llm_hass_api: str | None = None,
        user_llm_prompt: str | None = None,
    ) -> None:
        """Set the LLM system prompt."""
        llm_context = llm.LLMContext(
            platform=conversing_domain,
            context=user_input.context,
            user_prompt=user_input.text,
            language=user_input.language,
            assistant=DOMAIN,
            device_id=user_input.device_id,
        )

        llm_api: llm.APIInstance | None = None

        if user_llm_hass_api:
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    user_llm_hass_api,
                    llm_context,
                )
            except HomeAssistantError as err:
                LOGGER.error(
                    "Error getting LLM API %s for %s: %s",
                    user_llm_hass_api,
                    conversing_domain,
                    err,
                )
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "Error preparing LLM API",
                )
                raise ConverseError(
                    f"Error getting LLM API {user_llm_hass_api}",
                    conversation_id=self.conversation_id,
                    response=intent_response,
                ) from err

        user_name: str | None = None

        if (
            user_input.context
            and user_input.context.user_id
            and (
                user := await self.hass.auth.async_get_user(user_input.context.user_id)
            )
        ):
            user_name = user.name

        try:
            prompt_parts = [
                template.Template(
                    llm.BASE_PROMPT
                    + (user_llm_prompt or llm.DEFAULT_INSTRUCTIONS_PROMPT),
                    self.hass,
                ).async_render(
                    {
                        "ha_name": self.hass.config.location_name,
                        "user_name": user_name,
                        "llm_context": llm_context,
                    },
                    parse_result=False,
                )
            ]

        except TemplateError as err:
            LOGGER.error("Error rendering prompt: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I had a problem with my template",
            )
            raise ConverseError(
                "Error rendering prompt",
                conversation_id=self.conversation_id,
                response=intent_response,
            ) from err

        if llm_api:
            prompt_parts.append(llm_api.api_prompt)

        extra_system_prompt = (
            # Take new system prompt if one was given
            user_input.extra_system_prompt or self.extra_system_prompt
        )

        if extra_system_prompt:
            prompt_parts.append(extra_system_prompt)

        prompt = "\n".join(prompt_parts)

        self.llm_api = llm_api
        self.extra_system_prompt = extra_system_prompt
        self.content[0] = SystemContent(content=prompt)

        LOGGER.debug("Prompt: %s", self.content)
        LOGGER.debug("Tools: %s", self.llm_api.tools if self.llm_api else None)

        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {
                "messages": self.content,
                "tools": self.llm_api.tools if self.llm_api else None,
            },
        )
