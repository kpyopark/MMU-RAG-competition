"""
Context management for iterative section generation.

This module implements progressive context summarization with sliding window management
to maintain context budget while preserving key insights across section generation.

Key strategies:
- Compress previous sections to ≤200 tokens each (87% reduction)
- Sliding window: Recent 3-5 sections in full detail, older compressed
- Extract and maintain top 10 key insights across all sections
- Target: ≤40% of context window for efficient token usage
"""

from typing import List, Dict, Any
from loguru import logger

from .report_structure import GeneratedSection, ContextSummary
from .generator import get_llm_response


# Token estimation constants
TOKENS_PER_WORD = 1.3  # Average for English text
MAX_CONTEXT_BUDGET = 8000  # Target: 8K tokens of 20K window (40%)
SUMMARY_TARGET_TOKENS = 200  # Target tokens per section summary


COMPRESSION_PROMPT = """Compress the following report section into a concise summary of ≤200 tokens (~150 words).

**Section:** {section_title} ({section_id})
**Perspective:** {perspective}
**Word Count:** {word_count} words

**Full Content:**
{content}

**Instructions:**
1. Extract 3-5 key insights or findings
2. Preserve critical facts, numbers, and citations
3. Remove verbose explanations and redundant content
4. Maintain technical accuracy
5. Target length: 150 words (≤200 tokens)

**Compressed Summary:**"""


KEY_INSIGHTS_EXTRACTION_PROMPT = """Extract the top 10 most important insights from the following report sections.

**Report Sections:**
{sections_text}

**Instructions:**
1. Identify the 10 most critical findings, facts, or insights
2. Each insight should be 1-2 sentences
3. Prioritize unique, actionable, or high-impact information
4. Avoid redundancy between insights
5. Maintain factual accuracy

**Output Format:**
1. [First key insight]
2. [Second key insight]
...
10. [Tenth key insight]

**Top 10 Key Insights:**"""


class ContextManager:
    """
    Manages context window budget for iterative section generation.

    Uses progressive context summarization with sliding window to balance:
    - Maintaining rich context for coherent section generation
    - Staying within token budget to avoid context window overflow
    - Preserving key insights across all previous sections

    Token Economics:
    - Full section: ~500 tokens (350 words)
    - Compressed summary: ≤200 tokens (87% reduction)
    - Context budget: ≤8K tokens (40% of 20K Gemini window)
    """

    def __init__(self, sliding_window_size: int = 5):
        """
        Initialize context manager.

        Args:
            sliding_window_size: Number of recent sections to keep in full detail
                                Default: 5 (balances richness vs. budget)
        """
        self.sliding_window_size = sliding_window_size
        logger.info(
            f"Initialized ContextManager with sliding_window_size={sliding_window_size}"
        )

    def compress_section_to_summary(self, section: GeneratedSection) -> str:
        """
        Compress a section to ≤200 token summary.

        Args:
            section: Generated section with full content

        Returns:
            Compressed summary string (≤150 words)
        """
        logger.debug(
            f"Compressing section {section.get_section_id()} "
            f"({section.word_count} words → target ≤150 words)"
        )

        prompt = COMPRESSION_PROMPT.format(
            section_title=section.spec.title,
            section_id=section.get_section_id(),
            perspective=section.spec.perspective,
            word_count=section.word_count,
            content=section.content,
        )

        try:
            summary = get_llm_response(
                prompt=prompt,
                system_prompt="You are a concise summarization expert. Output summaries only.",
            )

            # Estimate token count
            estimated_tokens = int(len(summary.split()) * TOKENS_PER_WORD)

            logger.debug(
                f"Compressed {section.get_section_id()}: "
                f"{section.word_count} words → {len(summary.split())} words "
                f"(~{estimated_tokens} tokens, {100 * estimated_tokens / (section.word_count * TOKENS_PER_WORD):.0f}% of original)"
            )

            return summary

        except Exception as e:
            logger.warning(f"Compression failed for {section.get_section_id()}: {e}")
            # Fallback: simple truncation to first 150 words
            words = section.content.split()[:150]
            return " ".join(words) + "..."

    def build_generation_context(
        self,
        generated_sections: List[GeneratedSection],
        research_highlights: str,
        current_section_spec: Any = None,
    ) -> ContextSummary:
        """
        Build context for next section generation.

        Strategy:
        - Recent sections (last 3-5): Full detail for rich context
        - Older sections: Compressed summaries to save tokens
        - Key insights: Top 10 across all sections
        - Research highlights: Relevant Q&A from history

        Args:
            generated_sections: All previously generated sections
            research_highlights: Relevant excerpts from Q&A history
            current_section_spec: Spec for section being generated (optional)

        Returns:
            ContextSummary with compressed context within budget
        """
        if not generated_sections:
            # No previous sections, minimal context
            return ContextSummary(
                key_insights=[],
                previous_sections=[],
                research_highlights=research_highlights[:1000],  # Truncate
                total_tokens=self._estimate_tokens(research_highlights[:1000]),
            )

        logger.debug(
            f"Building context from {len(generated_sections)} previous sections "
            f"(sliding window: {self.sliding_window_size})"
        )

        # Sliding window: recent sections get full detail
        recent_sections = generated_sections[-self.sliding_window_size :]
        older_sections = generated_sections[: -self.sliding_window_size]

        # Build previous sections context
        previous_sections_text = []

        # Add older sections as summaries
        for section in older_sections:
            if section.summary:
                summary_text = f"[{section.get_section_id()}] {section.spec.title}: {section.summary}"
            else:
                # Fallback: compress on-the-fly
                summary = self.compress_section_to_summary(section)
                summary_text = f"[{section.get_section_id()}] {section.spec.title}: {summary}"

            previous_sections_text.append(summary_text)

        # Add recent sections in full detail
        for section in recent_sections:
            full_text = (
                f"[{section.get_section_id()}] {section.spec.title} (Full):\n{section.content}"
            )
            previous_sections_text.append(full_text)

        # Extract key insights across all sections
        key_insights = self._extract_key_insights(generated_sections)

        # Truncate research highlights if needed
        research_highlights_truncated = research_highlights[:2000]

        # Build context summary
        context = ContextSummary(
            key_insights=key_insights,
            previous_sections=previous_sections_text,
            research_highlights=research_highlights_truncated,
            total_tokens=self._estimate_context_tokens(
                key_insights, previous_sections_text, research_highlights_truncated
            ),
        )

        logger.info(
            f"Context built: {len(context.previous_sections)} sections "
            f"({len(older_sections)} compressed, {len(recent_sections)} full), "
            f"{len(context.key_insights)} key insights, "
            f"~{context.total_tokens} tokens "
            f"({100 * context.total_tokens / MAX_CONTEXT_BUDGET:.0f}% of budget)"
        )

        if not context.is_within_budget(MAX_CONTEXT_BUDGET):
            logger.warning(
                f"Context exceeds budget: {context.total_tokens} > {MAX_CONTEXT_BUDGET} tokens"
            )

        return context

    def _extract_key_insights(self, sections: List[GeneratedSection]) -> List[str]:
        """
        Extract top 10 key insights from all sections.

        Args:
            sections: All generated sections

        Returns:
            List of 10 key insights (strings)
        """
        if not sections:
            return []

        # Combine all section summaries (or full content if no summary)
        sections_text = []
        for section in sections:
            if section.summary:
                text = f"[{section.get_section_id()}] {section.spec.title}: {section.summary}"
            else:
                # Use first 200 words of content
                text = f"[{section.get_section_id()}] {section.spec.title}: {' '.join(section.content.split()[:200])}"
            sections_text.append(text)

        combined_text = "\n\n".join(sections_text)

        # Truncate if too long (max ~3000 words for insight extraction)
        if len(combined_text.split()) > 3000:
            combined_text = " ".join(combined_text.split()[:3000]) + "..."

        prompt = KEY_INSIGHTS_EXTRACTION_PROMPT.format(sections_text=combined_text)

        try:
            response = get_llm_response(
                prompt=prompt,
                system_prompt="You are an insight extraction expert. Output numbered lists only.",
            )

            # Parse numbered list
            insights = []
            for line in response.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove numbering prefix
                    insight = line.split(".", 1)[-1].strip() if "." in line else line
                    insight = insight.lstrip("- ").strip()
                    if insight:
                        insights.append(insight)

            # Limit to top 10
            insights = insights[:10]

            logger.debug(f"Extracted {len(insights)} key insights from {len(sections)} sections")

            return insights

        except Exception as e:
            logger.warning(f"Failed to extract key insights: {e}")
            # Fallback: return empty list
            return []

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using word count heuristic."""
        return int(len(text.split()) * TOKENS_PER_WORD)

    def _estimate_context_tokens(
        self, key_insights: List[str], previous_sections: List[str], research_highlights: str
    ) -> int:
        """Estimate total token count for context summary."""
        total_tokens = 0

        # Key insights
        total_tokens += sum(self._estimate_tokens(insight) for insight in key_insights)

        # Previous sections
        total_tokens += sum(
            self._estimate_tokens(section) for section in previous_sections
        )

        # Research highlights
        total_tokens += self._estimate_tokens(research_highlights)

        return total_tokens

    def format_context_for_prompt(self, context: ContextSummary) -> str:
        """
        Format context summary into prompt-ready string.

        Args:
            context: ContextSummary object

        Returns:
            Formatted string for inclusion in section generation prompt
        """
        parts = []

        # Key insights
        if context.key_insights:
            parts.append("**Key Insights from Previous Sections:**")
            for i, insight in enumerate(context.key_insights, 1):
                parts.append(f"{i}. {insight}")
            parts.append("")

        # Previous sections
        if context.previous_sections:
            parts.append("**Previous Sections:**")
            for section_text in context.previous_sections:
                parts.append(section_text)
                parts.append("")

        # Research highlights
        if context.research_highlights:
            parts.append("**Research Findings:**")
            parts.append(context.research_highlights)

        return "\n".join(parts)
