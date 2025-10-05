from typing import Callable, Dict, Any, List
from .retriever import retrieve
from .generator import get_llm_response, self_evolve
from loguru import logger


NUM_VARIANTS = 2  # 2
EVOLUTION_STEPS = 1
MAX_SEARCH_ITERATIONS = 3  # 3
SEARCH_TOP_K = 10

PLAN_PROMPT = """
Based on the user's query, create a structured research plan.
This plan should outline the key areas, questions, and topics to investigate to provide a comprehensive answer.
The plan will serve as a scaffold for the entire research process.
Break it down into a list of concise points.

User Query: "{query}"
"""

INITIAL_DRAFT_PROMPT = """
Based on your internal knowledge and the user's query, write a preliminary, high-level draft report.
This draft will be refined later with retrieved information. It serves as a starting point and a "noisy" skeleton.

User Query: "{query}"
"""

SEARCH_QUERY_GEN_PROMPT = """
You are a researcher in an iterative process. Your goal is to formulate the next best search query to gather information to refine an evolving research report.

**User's Original Query:**
{query}

**Overall Research Plan:**
{plan}

**Current Draft Report (State to be improved):**
{draft}

**History of Previous Searches (Queries and Answers):**
{history}

Based on all the above information, what is the single most important search query to execute right now?
The query should be concise, targeted, and aimed at filling gaps or verifying information in the current draft.
Do not ask a question that has already been answered in the history.
Output only the search query, with no preamble.
"""

ANSWER_SYNTHESIS_PROMPT = """
You are a research analyst. You have been given a search query and a list of retrieved documents.
Your task is to synthesize the information from these documents to provide a direct and comprehensive answer to the search query.
Focus only on the information present in the documents. Cite which document urls are relevant.

**Search Query:**
{search_query}

**Retrieved Document Chunks:**
{documents}

Synthesized Answer:
"""

DRAFT_REVISION_PROMPT = """
You are refining a research report. You have a previous version of the draft and new information from a recent search.
Your task is to integrate the new information into the draft to "denoise" it, making it more accurate, detailed, and comprehensive.
You can add new sections, expand existing points, or correct inaccuracies.

**User's Original Query:**
{query}

**Previous Draft Report:**
---
{draft}
---

**Newly Synthesized Information (from query: "{search_query}"):**
---
{new_answer}
---

Produce the new, revised draft report.
"""

FINAL_REPORT_PROMPT = """
You are a research assistant tasked with writing a final, comprehensive report.
All the necessary research, including planning, iterative searching, and information synthesis, has been completed.
Use all the provided information to construct a well-structured, coherent, and detailed final report that directly addresses the user's original query.

**User's Original Query:**
{query}

**Initial Research Plan:**
{plan}

**Final Revised Draft (Skeleton for the report):**
{draft}

**Full History of Questions and Synthesized Answers:**
{history}

Now, write the final, polished report. Start with a "Final Answer:" short paragraph summarizing the key findings, followed by detailed sections below and citations where relevant.
"""


class TTD_DR_Pipeline:
    def __init__(self, callback: Callable[[Dict[str, Any]], None]):
        self.callback = callback
        self.plan = ""
        self.draft = ""
        self.q_a_history: List[Dict[str, str]] = []

    def _send_update(
        self,
        step_description: str,
        is_intermediate: bool = True,
        final_report_chunk: str = "",
        citations: List = [],
        complete: bool = False,
    ):
        """Helper to send updates through the callback."""
        # Join all previous steps with the new one
        all_steps = [s["description"] for s in self.q_a_history]
        if step_description:
            all_steps.append(step_description)

        data = {
            "intermediate_steps": "|||---|||".join(all_steps),
            "final_report": final_report_chunk,
            "is_intermediate": is_intermediate,
            "citations": citations,
            "complete": complete,
        }
        self.callback(data)

    def generate_research_plan(self, query: str):
        """Generate initial research plan based on user query."""
        step_desc = "Generating initial research plan..."
        self._send_update(step_desc)
        plan_prompt = PLAN_PROMPT.format(query=query)
        self.plan = self_evolve(
            plan_prompt,
            "You are a strategic research planner.",
            num_variants=NUM_VARIANTS,
            evolution_steps=EVOLUTION_STEPS,
        )
        self.q_a_history.append({"description": f"**Research Plan:**\n{self.plan}"})
        self._send_update("")

    def generate_initial_draft(self, query: str):
        """Generate initial draft from internal knowledge."""
        self._send_update("Generating initial draft from internal knowledge...")
        draft_prompt = INITIAL_DRAFT_PROMPT.format(query=query)
        self.draft = get_llm_response(draft_prompt)
        self.q_a_history.append(
            {"description": f"**Initial Draft:**\n{self.draft[:200]}..."}
        )
        self._send_update("")

    def generate_search_query(self, query: str, iteration: int, max_iterations: int):
        """Generate next search query for the current iteration."""
        step_desc = f"**Iteration {iteration + 1}/{max_iterations}:** Generating next search query..."
        self._send_update(step_desc)
        history_str = "\n".join(
            [
                f"Q: {item['query']}\nA: {item['answer']}"
                for item in self.q_a_history
                if "query" in item
            ]
        )
        search_gen_prompt = SEARCH_QUERY_GEN_PROMPT.format(
            query=query, plan=self.plan, draft=self.draft, history=history_str
        )
        # TODO: try to generate several queries and rank the results to save tokens
        search_query = get_llm_response(search_gen_prompt)
        self._send_update(f"**Searching for:** `{search_query}`")
        return search_query

    def retrieve_and_synthesize_documents(self, search_query: str):
        chunks = retrieve(search_query, top_k=SEARCH_TOP_K)
        doc_str = "\n\n".join(
            [
                f"ID: {doc['chunk_id']}\nText: {doc['text']}..."
                for doc in chunks
            ]
        )
        citations = [doc.get("url") for doc in chunks]
        self._send_update(
            f"**Found {len(chunks)} documents.** Synthesizing answer...",
            citations=citations,
        )

        synth_prompt = ANSWER_SYNTHESIS_PROMPT.format(
            search_query=search_query, documents=doc_str
        )
        # TODO: log variants and use citations
        synthesized_answer, variants = self_evolve(
            synth_prompt,
            "You are a research analyst.",
            num_variants=NUM_VARIANTS,
            evolution_steps=EVOLUTION_STEPS,
        )
        self.q_a_history.append(
            {
                "description": f"**Synthesized Answer for `{search_query}`:**\n{synthesized_answer}",
                "query": search_query,
                "answer": synthesized_answer,
            }
        )
        self._send_update("")
        return synthesized_answer

    def revise_draft_with_new_info(
        self, query: str, search_query: str, synthesized_answer: str, iteration: int
    ):
        """Revise the current draft with new synthesized information."""
        step_desc = "Revising draft with new information..."
        self._send_update(step_desc)
        revise_prompt = DRAFT_REVISION_PROMPT.format(
            query=query,
            draft=self.draft,
            search_query=search_query,
            new_answer=synthesized_answer,
        )
        self.draft = get_llm_response(revise_prompt)
        self.q_a_history.append(
            {
                "description": f"**Revised Draft {iteration + 1}:**\n{self.draft[:200]}..."
            }
        )
        self._send_update("")

    def perform_iterative_search_and_synthesis(self, query: str, max_iterations: int):
        """Perform iterative search and synthesis loop to refine the draft."""
        for i in range(max_iterations):
            search_query = self.generate_search_query(query, i, max_iterations)
            if search_query is None or search_query.strip() == "":
                logger.warning("No valid search query generated, stopping iterations.")
                continue
            synthesized_answer = self.retrieve_and_synthesize_documents(search_query)
            self.revise_draft_with_new_info(query, search_query, synthesized_answer, i)

    def generate_final_report(self, query: str):
        self._send_update("All research steps complete. Generating final report...")
        history_str = "\n\n".join(
            [
                f"**Question:** {item['query']}\n**Answer:** {item['answer']}"
                for item in self.q_a_history
                if "query" in item
            ]
        )
        final_prompt = FINAL_REPORT_PROMPT.format(
            query=query, plan=self.plan, draft=self.draft, history=history_str
        )

        final_report_content = get_llm_response(final_prompt)
        self._send_update(
            "Final report generated.",
            is_intermediate=False,
            final_report_chunk=final_report_content,
            complete=True,
        )

    def run(self, query: str, max_iterations: int = MAX_SEARCH_ITERATIONS):
        self.generate_research_plan(query)
        self.generate_initial_draft(query)
        self.perform_iterative_search_and_synthesis(query, max_iterations)
        self.generate_final_report(query)


def run_rag_dynamic(query: str, callback: Callable[[Dict[str, Any]], None]):
    """Entry point for the streaming RAG pipeline."""
    pipeline = TTD_DR_Pipeline(callback)
    pipeline.run(query)


def run_rag_static(query: str) -> str:
    """Entry point for the static RAG pipeline, returns the final report as a string."""
    final_report = ""

    def static_callback(data: Dict[str, Any]):
        nonlocal final_report
        if data["complete"]:
            final_report = data["final_report"]
        print(f"Static Callback Update: {data}")

    pipeline = TTD_DR_Pipeline(static_callback)
    pipeline.run(query)
    return final_report
