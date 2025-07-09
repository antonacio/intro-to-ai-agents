from datetime import datetime
from textwrap import dedent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from examples.config import MODEL_PROVIDER, llm, ModelProviders


@tool(
    response_format="content_and_artifact",
    # add today's date to the description to prevent the LLM to use their training cutoff date in the tool call
    description=dedent(
        """\
        Searches the web for current information and real-time data.

        Use this tool when you need current events or news, recent developments, or real-time data (stock prices, weather, sports scores, etc.)

        Query Guidelines:
        - Use specific, focused search terms. Avoid overly broad or vague queries
        - Include relevant context (dates, locations, specific entities)
        - For reference, today's date is {todays_date}
        - Use the same language as the user's question

        Args:
            query: A specific web search query.
        """
    ).format(todays_date=datetime.today().strftime("%B %d, %Y (%Y-%m-%d)")),
)
def web_search(query: str) -> tuple[dict, AIMessage]:
    if MODEL_PROVIDER != ModelProviders.OPENAI:
        raise ValueError(
            f"Unsupported model provider for web search: '{MODEL_PROVIDER}'. "
            f"Only '{ModelProviders.OPENAI}' is supported for web search."
        )

    tool = {"type": "web_search_preview"}
    llm_with_web_search = llm.bind_tools([tool])

    response = llm_with_web_search.invoke(query)

    return response.content[0]["text"], response
