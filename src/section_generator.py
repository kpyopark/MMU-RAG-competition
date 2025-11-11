"""
Section generation component for iterative report synthesis.

This module generates individual report sections (300-500 words) while managing
output token limits (≤2048 tokens) and maintaining coherence with previous sections.

Key features:
- Generate sections with rich context from previous work
- Stay within Gemini API output token limit (2048)
- Extract inline citations and track usage
- Generate executive summary and conclusion sections
"""

import time
import re
from typing import List
from loguru import logger

from .report_structure import SectionSpec, GeneratedSection, ContextSummary, ReportStructure
from .generator import get_llm_response


SECTION_GENERATION_PROMPT = """You are writing a specific section of a comprehensive research report.

**Current Section:** {section_title} (Section {section_id})
**Chapter:** {chapter_title}
**Perspective:** {perspective}
**Target Length:** {target_word_count} words

**Section Guidance:**
{guidance}

**Context from Previous Work:**
{context_summary}

**Research Data Available:**
{research_data}

**Instructions:**
1. Write a detailed, well-researched section of {target_word_count} words
2. Build on insights from previous sections (avoid redundancy)
3. Use inline citations in format [Source N] for all factual claims
4. Provide specific details, data, and analysis
5. Maintain coherent narrative flow with previous sections
6. Stay within {max_output_tokens} output tokens

**Write the section now:**"""


EXECUTIVE_SUMMARY_PROMPT = """Write a comprehensive Executive Summary for the following research report.

**User Query:**
{query}

**Report Structure:**
{report_outline}

**Research Data:**
{research_data}

**Instructions:**
1. Provide high-level synthesis covering all major perspectives
2. Highlight 3-5 key findings across all chapters
3. Target length: 400 words
4. Include inline citations [Source N] for major claims
5. Set clear expectations for what the report covers

**Executive Summary:**"""


CONCLUSION_PROMPT = """Write a comprehensive Conclusion for the following research report.

**User Query:**
{query}

**Report Sections Summary:**
{sections_summary}

**Instructions:**
1. Synthesize findings from all previous sections
2. Provide forward-looking implications and recommendations
3. Discuss potential future developments or scenarios
4. Target length: 400 words
5. Include inline citations [Source N] where appropriate
6. End with clear takeaways

**Conclusion:**"""


class SectionGenerator:
    """
    Generates individual report sections with context awareness.

    Manages:
    - Section-by-section generation (300-500 words each)
    - Output token limit compliance (≤2048 tokens)
    - Context integration from previous sections
    - Citation extraction and tracking
    - Special section types (executive summary, conclusion)
    """

    def __init__(self):
        """Initialize section generator."""
        logger.info("Initialized SectionGenerator")

    def generate_section(
        self,
        spec: SectionSpec,
        context_summary: ContextSummary,
        research_data: str,
        regeneration_guidance: str = "",
    ) -> GeneratedSection:
        """
        Generate a single report section.

        Args:
            spec: Section specification with title, perspective, guidance
            context_summary: Compressed context from previous sections
            research_data: Relevant research findings from Q&A history
            regeneration_guidance: Additional guidance for regeneration attempts

        Returns:
            GeneratedSection with content, metadata, citations
        """
        start_time = time.time()

        logger.info(
            f"Generating section {spec.get_full_id()}: {spec.title} "
            f"(target: {spec.target_word_count} words, perspective: {spec.perspective})"
        )

        # Format context
        from .context_manager import ContextManager

        context_mgr = ContextManager()
        context_text = context_mgr.format_context_for_prompt(context_summary)

        # Build prompt
        prompt = SECTION_GENERATION_PROMPT.format(
            section_title=spec.title,
            section_id=spec.get_full_id(),
            chapter_title=f"Chapter {spec.chapter_number}",
            perspective=spec.perspective,
            target_word_count=spec.target_word_count,
            guidance=spec.guidance
            + (f"\n\nREGENERATION GUIDANCE:\n{regeneration_guidance}" if regeneration_guidance else ""),
            context_summary=context_text,
            research_data=research_data[:3000],  # Truncate if too long
            max_output_tokens=spec.max_output_tokens,
        )

        try:
            # Generate section with output token limit
            content = get_llm_response(
                prompt=prompt,
                system_prompt="You are a detailed research report writer. Write comprehensive, well-cited sections.",
            )

            # Extract metadata
            word_count = len(content.split())
            citations = self._extract_citations(content)
            generation_time = time.time() - start_time

            logger.info(
                f"Generated section {spec.get_full_id()}: "
                f"{word_count} words, {len(citations)} citations, "
                f"{generation_time:.1f}s"
            )

            return GeneratedSection(
                spec=spec,
                content=content,
                word_count=word_count,
                citations_used=citations,
                generation_time=generation_time,
                summary="",  # Will be compressed later
            )

        except Exception as e:
            logger.error(f"Failed to generate section {spec.get_full_id()}: {e}")
            # Return minimal fallback section
            fallback_content = (
                f"# {spec.title}\n\n"
                f"[Content generation failed for this section. Error: {str(e)}]\n\n"
                f"This section was intended to cover: {spec.guidance}"
            )
            return GeneratedSection(
                spec=spec,
                content=fallback_content,
                word_count=len(fallback_content.split()),
                citations_used=[],
                generation_time=time.time() - start_time,
                summary="",
            )

    def generate_executive_summary(
        self, structure: ReportStructure, query: str, research_data: str
    ) -> GeneratedSection:
        """
        Generate Executive Summary section.

        Args:
            structure: Complete report structure with all chapters
            query: Original user query
            research_data: Research findings from Q&A history

        Returns:
            GeneratedSection for executive summary
        """
        start_time = time.time()

        logger.info("Generating Executive Summary")

        # Format report outline
        report_outline = self._format_report_outline(structure)

        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            query=query,
            report_outline=report_outline,
            research_data=research_data[:3000],
        )

        try:
            content = get_llm_response(
                prompt=prompt,
                system_prompt="You are an executive summary writer. Provide clear, high-level syntheses.",
            )

            word_count = len(content.split())
            citations = self._extract_citations(content)
            generation_time = time.time() - start_time

            logger.info(
                f"Generated Executive Summary: {word_count} words, "
                f"{len(citations)} citations, {generation_time:.1f}s"
            )

            return GeneratedSection(
                spec=structure.executive_summary,
                content=content,
                word_count=word_count,
                citations_used=citations,
                generation_time=generation_time,
                summary="",
            )

        except Exception as e:
            logger.error(f"Failed to generate Executive Summary: {e}")
            fallback_content = (
                f"# Executive Summary\n\n"
                f"[Executive summary generation failed. Error: {str(e)}]"
            )
            return GeneratedSection(
                spec=structure.executive_summary,
                content=fallback_content,
                word_count=len(fallback_content.split()),
                citations_used=[],
                generation_time=time.time() - start_time,
                summary="",
            )

    def generate_conclusion(
        self, structure: ReportStructure, sections: List[GeneratedSection], query: str
    ) -> GeneratedSection:
        """
        Generate Conclusion section.

        Args:
            structure: Complete report structure
            sections: All previously generated sections
            query: Original user query

        Returns:
            GeneratedSection for conclusion
        """
        start_time = time.time()

        logger.info("Generating Conclusion")

        # Build sections summary
        sections_summary = self._build_sections_summary(sections)

        prompt = CONCLUSION_PROMPT.format(
            query=query, sections_summary=sections_summary
        )

        try:
            content = get_llm_response(
                prompt=prompt,
                system_prompt="You are a report conclusion writer. Synthesize findings and provide forward-looking analysis.",
            )

            word_count = len(content.split())
            citations = self._extract_citations(content)
            generation_time = time.time() - start_time

            logger.info(
                f"Generated Conclusion: {word_count} words, "
                f"{len(citations)} citations, {generation_time:.1f}s"
            )

            return GeneratedSection(
                spec=structure.conclusion,
                content=content,
                word_count=word_count,
                citations_used=citations,
                generation_time=generation_time,
                summary="",
            )

        except Exception as e:
            logger.error(f"Failed to generate Conclusion: {e}")
            fallback_content = f"# Conclusion\n\n[Conclusion generation failed. Error: {str(e)}]"
            return GeneratedSection(
                spec=structure.conclusion,
                content=fallback_content,
                word_count=len(fallback_content.split()),
                citations_used=[],
                generation_time=time.time() - start_time,
                summary="",
            )

    def _extract_citations(self, content: str) -> List[str]:
        """
        Extract citation references from content.

        Looks for [Source N] or [N] patterns and extracts unique citation numbers.

        Args:
            content: Generated section content

        Returns:
            List of citation markers (e.g., ["Source 1", "Source 3"])
        """
        # Pattern: [Source N] or [N]
        pattern = r'\[(?:Source\s+)?(\d+)\]'
        matches = re.findall(pattern, content)

        # Convert to set to remove duplicates, then to sorted list
        unique_citations = sorted(set(f"Source {n}" for n in matches))

        return unique_citations

    def _format_report_outline(self, structure: ReportStructure) -> str:
        """Format report structure as outline string."""
        lines = [f"Total Sections: {structure.total_sections()}\n"]

        for chapter in structure.chapters:
            lines.append(
                f"\nChapter {chapter.chapter_number}: {chapter.title} ({chapter.perspective})"
            )
            for section in chapter.sections:
                lines.append(f"  - Section {section.get_full_id()}: {section.title}")

        return "\n".join(lines)

    def _build_sections_summary(self, sections: List[GeneratedSection]) -> str:
        """Build summary of all sections for conclusion prompt."""
        summaries = []

        for section in sections:
            # Use summary if available, otherwise first 100 words of content
            if section.summary:
                summary_text = section.summary
            else:
                summary_text = " ".join(section.content.split()[:100])

            summaries.append(
                f"[{section.get_section_id()}] {section.spec.title}:\n{summary_text}"
            )

        return "\n\n".join(summaries)
