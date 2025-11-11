from typing import Callable, Dict, Any, List
from .retriever import retrieve, retrieve_with_grounded_generation
from .generator import get_llm_response, self_evolve
from .structure_generator import ReportStructureGenerator
from .context_manager import ContextManager
from .section_generator import SectionGenerator
from .quality_validator import QualityValidator
from .report_assembler import ReportAssembler
from .report_structure import GeneratedSection, ReportStructure
from loguru import logger


NUM_VARIANTS = 1  # 2
EVOLUTION_STEPS = 1
MAX_SEARCH_ITERATIONS = 1  # 3
SEARCH_TOP_K = 50

# Feature flag for structured report generation
ENABLE_STRUCTURED_REPORTS = True

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
You have been given a search query and a list of retrieved documents.
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

**Citations:**
{citations}

Now, write the final, polished report. Start with a "Final Answer:" short paragraph summarizing the key findings, followed by detailed sections below and citations where relevant.
"""


class TTD_DR_Pipeline:
    def __init__(self, callback: Callable[[Dict[str, Any]], None], enable_structured: bool = ENABLE_STRUCTURED_REPORTS):
        self.callback = callback
        self.plan = ""
        self.draft = ""
        self.q_a_history: List[Dict[str, str]] = []
        self.intermediate_log: List[str] = []
        self.citations: List[str] = []

        # Structured report generation components
        self.enable_structured = enable_structured
        self.report_structure: ReportStructure | None = None
        self.generated_sections: List[GeneratedSection] = []

        if self.enable_structured:
            self.structure_generator = ReportStructureGenerator()
            self.context_manager = ContextManager()
            self.section_generator = SectionGenerator()
            self.quality_validator = QualityValidator()
            self.report_assembler = ReportAssembler()
            logger.info("Structured report generation ENABLED")

    def _send_update(
        self,
        step_description: str | None = None,
        *,
        is_intermediate: bool = True,
        final_report_chunk: str | None = None,
        citations: List[str] | None = None,
        complete: bool = False,
    ) -> None:
        """Helper to send updates through the callback."""
        if step_description:
            self.intermediate_log.append(step_description)

        steps_text = "|||---|||".join(self.intermediate_log)
        data: Dict[str, Any] = {
            "intermediate_steps": steps_text if steps_text else None,
            "final_report": final_report_chunk,
            "is_intermediate": is_intermediate,
            "complete": complete,
        }
        if citations:
            data["citations"] = citations

        self.callback(data)

    def generate_research_plan(self, query: str):
        """Generate initial research plan based on user query."""
        self._send_update("Generating initial research plan...")
        plan_prompt = PLAN_PROMPT.format(query=query)
        plan_text, _ = self_evolve(
            plan_prompt,
            "You are a strategic research planner.",
            num_variants=NUM_VARIANTS,
            evolution_steps=EVOLUTION_STEPS,
        )
        self.plan = plan_text
        plan_desc = f"**Research Plan Generated:**\n{self.plan}"
        self.q_a_history.append({"description": plan_desc})
        self._send_update(plan_desc)

    def generate_initial_draft(self, query: str):
        """Generate initial draft from internal knowledge."""
        self._send_update("Generating initial draft from internal knowledge...")
        draft_prompt = INITIAL_DRAFT_PROMPT.format(query=query)
        self.draft = get_llm_response(draft_prompt)
        draft_desc = f"**Initial Draft Created:**\n{self.draft[:200]}..."
        self.q_a_history.append({"description": draft_desc})
        self._send_update(draft_desc)

    def generate_search_query(self, query: str, iteration: int, max_iterations: int):
        """Generate next search query for the current iteration."""
        step_desc = f"**Iteration {iteration + 1}/{max_iterations}:** Generating next search query..."
        self._send_update(step_desc)

        # QUICK FIX: For first iteration, use the original user query directly
        # This avoids LLM bias from an incorrect initial draft
        # See INTEGRATION_REPORT.md for root cause analysis
        if iteration == 0:
            search_query = query
            self._send_update(f"**Searching for (direct query):** `{search_query}`")
            return search_query

        # For subsequent iterations, use LLM to generate targeted queries
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
        self._send_update(f"**Searching for (generated query):** `{search_query}`")
        return search_query

    def retrieve_and_synthesize_documents(self, search_query: str):
        """
        Retrieve and synthesize using Gemini's grounded generation.

        This replaces the old Search → Chunk → Rerank → Synthesize pipeline
        with a single Gemini API call that automatically:
        - Searches Google for relevant information
        - Generates comprehensive answer
        - Returns citations
        """
        self._send_update("Searching web and synthesizing answer with Gemini grounded generation...")

        try:
            # Use Gemini's grounded generation (single API call)
            synthesized_answer, citations_list = retrieve_with_grounded_generation(
                query=search_query,
                context_prompt=f"""You are researching to answer this query: {search_query}

Provide a comprehensive, well-researched answer based on current web information.
Focus on specific facts, data, and details from authoritative sources."""
            )

            # Extract citation URLs
            citations = [cit["url"] for cit in citations_list]
            self.citations.extend(citations)

            self._send_update(
                f"**Grounded generation complete:** {len(citations)} sources used. Synthesizing answer...",
                citations=citations if citations else None,
            )

            # Add to Q&A history
            self.q_a_history.append(
                {
                    "description": f"**Synthesized Answer for `{search_query}`:**\n{synthesized_answer}",
                    "query": search_query,
                    "answer": synthesized_answer,
                }
            )
            self._send_update(
                f"**Synthesized Answer for `{search_query}`:**\n{synthesized_answer}"
            )

            return synthesized_answer

        except Exception as e:
            logger.error(f"Error in grounded generation: {e}")
            # Fallback: return error message but don't crash the pipeline
            error_msg = f"Unable to retrieve web information for this query: {str(e)}"
            self._send_update(f"**Warning:** {error_msg}")
            return error_msg

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
        revised_desc = f"**Revised Draft {iteration + 1}:**\n{self.draft[:200]}..."
        self.q_a_history.append({"description": revised_desc})
        self._send_update(revised_desc)

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
        citations_str = "\n".join(self.citations)
        final_prompt = FINAL_REPORT_PROMPT.format(
            query=query,
            plan=self.plan,
            draft=self.draft,
            history=history_str,
            citations=citations_str,
        )

        final_report_content = get_llm_response(final_prompt)

        self._send_update(
            "Final report generated.",
            is_intermediate=False,
            final_report_chunk=final_report_content,
            citations=self.citations,
            complete=True,
        )

    def generate_report_structure(self, query: str):
        """Generate comprehensive report structure from research context."""
        if not self.enable_structured:
            return

        self._send_update("Generating comprehensive report structure...")

        # Prepare research summary from Q&A history
        research_summary = "\n".join(
            [
                f"Q: {item['query']}\nA: {item['answer'][:200]}..."
                for item in self.q_a_history
                if "query" in item
            ]
        )

        # Generate structure
        self.report_structure = self.structure_generator.generate_chapter_outline(
            query=query, plan=self.plan, research_summary=research_summary
        )

        # Send structure update
        structure_desc = (
            f"**Report Structure Generated:**\n"
            f"- {self.report_structure.total_sections()} total sections\n"
            f"- {len(self.report_structure.chapters)} chapters\n"
            f"- ~{self.report_structure.estimated_word_count} target words"
        )
        self._send_update(structure_desc)

    def generate_structured_report(self, query: str):
        """Generate report through iterative section-by-section synthesis."""
        if not self.enable_structured or not self.report_structure:
            # Fallback to legacy single-pass generation
            self.generate_final_report(query)
            return

        self._send_update("Starting structured report generation...")

        # Prepare research data for sections
        research_data = self._prepare_research_data()

        # 1. Generate executive summary
        exec_summary = self.section_generator.generate_executive_summary(
            self.report_structure, query, research_data
        )
        self.generated_sections.append(exec_summary)
        self._send_update(
            f"Executive Summary generated ({exec_summary.word_count} words)"
        )

        # 2. Iterative section generation
        total_main_sections = sum(
            len(ch.sections) for ch in self.report_structure.chapters
        )
        current_section = 0

        for chapter in self.report_structure.chapters:
            self._send_update(f"Starting Chapter {chapter.chapter_number}: {chapter.title}")

            for section_spec in chapter.sections:
                current_section += 1
                progress = f"Section {current_section}/{total_main_sections}"

                # Generate section with validation
                section = self._generate_section_with_validation(
                    section_spec, research_data, progress
                )

                # Compress to summary for context
                section.summary = self.context_manager.compress_section_to_summary(
                    section
                )
                self.generated_sections.append(section)

                self._send_update(
                    f"Completed {progress}: {section.spec.title} "
                    f"({section.word_count} words, {len(section.citations_used)} citations)"
                )

        # 3. Generate conclusion
        conclusion = self.section_generator.generate_conclusion(
            self.report_structure, self.generated_sections, query
        )
        self.generated_sections.append(conclusion)
        self._send_update(f"Conclusion generated ({conclusion.word_count} words)")

        # 4. Assemble final report
        self._send_update("Assembling final report...")
        final_report = self.report_assembler.assemble_final_report(
            self.report_structure, self.generated_sections
        )

        # Send final report
        self._send_update(
            "Structured report generation complete.",
            is_intermediate=False,
            final_report_chunk=final_report,
            citations=self.citations,
            complete=True,
        )

    def _prepare_research_data(self) -> str:
        """Prepare research findings for section generation."""
        research_parts = []

        for item in self.q_a_history:
            if "query" in item:
                research_parts.append(
                    f"**Research Query:** {item['query']}\n"
                    f"**Findings:** {item['answer']}\n"
                )

        return "\n".join(research_parts)

    def _generate_section_with_validation(
        self, section_spec, research_data: str, progress: str
    ) -> GeneratedSection:
        """Generate section with quality validation and regeneration."""
        max_attempts = 2
        attempt = 1

        while attempt <= max_attempts:
            # Build context
            context = self.context_manager.build_generation_context(
                self.generated_sections, research_data
            )

            # Generate section
            regeneration_guidance = ""
            if attempt > 1:
                self._send_update(
                    f"{progress}: Regenerating (attempt {attempt}/{max_attempts})..."
                )

            section = self.section_generator.generate_section(
                spec=section_spec,
                context_summary=context,
                research_data=research_data,
                regeneration_guidance=regeneration_guidance,
            )

            # Validate
            validation_result = self.quality_validator.validate_section(
                section, self.generated_sections, attempt
            )

            should_regen, guidance = self.quality_validator.should_regenerate(
                validation_result, attempt, max_attempts
            )

            if not should_regen:
                # Validation passed or max attempts reached
                if not validation_result.is_valid:
                    self._send_update(
                        f"⚠️ {progress}: Quality issues detected but proceeding (max attempts reached)"
                    )
                return section

            # Regeneration needed
            regeneration_guidance = guidance
            attempt += 1

        return section

    def run(self, query: str, max_iterations: int = MAX_SEARCH_ITERATIONS):
        self.generate_research_plan(query)
        self.generate_initial_draft(query)
        self.perform_iterative_search_and_synthesis(query, max_iterations)

        # Choose report generation mode
        if self.enable_structured:
            self.generate_report_structure(query)
            self.generate_structured_report(query)
        else:
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
        logger.debug(f"Static callback update: {data}")

    pipeline = TTD_DR_Pipeline(static_callback)
    pipeline.run(query)
    return final_report
