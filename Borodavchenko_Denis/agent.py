
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], "The messages to send to the LLM", operator.add]
    report: Annotated[str, "The report about the summaries", operator.add]

class Agent:
    def __init__(self, llm, tools, system_message: str):
        self.system_message = system_message
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.call_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_node("generate_report", self.generate_report)
        graph.add_edge("action", "generate_report")
        graph.add_conditional_edges(
            "action",
            self.should_generate_report,
            {True: "generate_report", False: "llm"}
        )
        graph.set_entry_point("llm")
        
        self.graph = graph.compile()
        self.tools = {t.__name__: t for t in tools}
        self.llm = llm.bind_tools(tools)


    def call_llm(self, state: AgentState):
        messages = state["messages"]
        if self.system_message:
            messages = [SystemMessage(content=self.system_message)] + messages
        message = self.llm.invoke(messages)
        return {"messages": [message]}
    
    def call_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            if not t["name"] in self.tools:
                print(f"Tool {t['name']} not found")
                result = "Error: Tool not found"
            else:
                result = self.tools[t['name']](t['args']['text'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            
        return {"messages": results}
    
    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0
    
    def should_generate_report(self, state: AgentState):
        tool_results = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
        tool_names = {msg.name for msg in tool_results}
        return "extractive_summarization" in tool_names and "abstractive_summarization" in tool_names

    def generate_report(self, state: AgentState):
        tool_results = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
        
        # Group results by tool type
        extractive_results = [msg.content for msg in tool_results if msg.name == "extractive_summarization"]
        abstractive_results = [msg.content for msg in tool_results if msg.name == "abstractive_summarization"]
        
        # Create context for the LLM to generate a report
        report_prompt = f"""
        Compare and analyze the following summaries:
        
        Extractive Summary: {extractive_results[0] if extractive_results else "N/A"}
        
        Abstractive Summary: {abstractive_results[0] if abstractive_results else "N/A"}
        
        Generate a detailed report analyzing both approaches.
        Include both the extractive and abstractive summaries in the report so we can see the difference.
        """
        
        # Use the LLM to generate the report
        report = self.llm.invoke([HumanMessage(content=report_prompt)])
        
        return {"report": report.content}
