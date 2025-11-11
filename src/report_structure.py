"""
Data models for structured report generation.

This module defines the core data structures used in the iterative section-by-section
report generation system, including:
- SectionSpec: Specification for a section to be generated
- Chapter: Logical grouping of sections by perspective
- ReportStructure: Overall report outline with chapters and sections
- GeneratedSection: Section content with metadata after generation
- ContextSummary: Compressed context for subsequent section generation
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class SectionSpec:
    """
    Specification for a single section to be generated.

    Each section represents a focused analytical unit within a chapter,
    targeting 300-500 words with specific perspective and guidance.
    """
    title: str
    chapter_number: int
    section_number: int
    perspective: str  # e.g., "Financial", "Technical", "Regulatory"
    guidance: str  # Detailed instructions for section content
    target_word_count: int = 350
    max_output_tokens: int = 2048  # Gemini API constraint

    def get_full_id(self) -> str:
        """Return unique identifier for this section (e.g., '2.3')."""
        return f"{self.chapter_number}.{self.section_number}"


@dataclass
class Chapter:
    """
    Logical grouping of sections by analytical perspective.

    Each chapter focuses on one major perspective (e.g., Financial Analysis,
    Strategic Implications) and contains 3-5 detailed sections.
    """
    title: str
    perspective: str  # Primary analytical lens for this chapter
    sections: List[SectionSpec]
    chapter_number: int

    def total_target_words(self) -> int:
        """Calculate total target word count for all sections in this chapter."""
        return sum(section.target_word_count for section in self.sections)


@dataclass
class ReportStructure:
    """
    Complete outline for a structured multi-chapter report.

    Represents the hierarchical structure of the final report:
    - Executive Summary (1 section)
    - Main Chapters (4-6 chapters, each with 3-5 sections)
    - Conclusion (1 section)
    """
    executive_summary: SectionSpec
    chapters: List[Chapter]
    conclusion: SectionSpec
    estimated_word_count: int
    estimated_sections: int
    created_at: datetime = field(default_factory=datetime.utcnow)

    def total_sections(self) -> int:
        """Return total number of sections including executive summary and conclusion."""
        return 2 + sum(len(ch.sections) for ch in self.chapters)

    def get_all_sections(self) -> List[SectionSpec]:
        """Return ordered list of all sections in generation order."""
        sections = [self.executive_summary]
        for chapter in self.chapters:
            sections.extend(chapter.sections)
        sections.append(self.conclusion)
        return sections


@dataclass
class GeneratedSection:
    """
    Section content and metadata after generation.

    Stores the generated text, citations, quality metrics, and a compressed
    summary for use as context in subsequent section generation.
    """
    spec: SectionSpec
    content: str
    word_count: int
    citations_used: List[str]  # URLs cited in this section
    generation_time: float  # Seconds
    summary: str = ""  # Compressed to ≤200 tokens for context propagation

    def get_section_id(self) -> str:
        """Return unique identifier for this section (e.g., '2.3')."""
        return self.spec.get_full_id()

    def citation_density(self) -> float:
        """Calculate citations per 150 words (target: ≥1.0)."""
        if self.word_count == 0:
            return 0.0
        return (len(self.citations_used) / self.word_count) * 150


@dataclass
class ContextSummary:
    """
    Compressed context for section generation.

    Maintains context window budget by:
    - Limiting key insights to top 10
    - Compressing previous sections to ≤200 tokens each
    - Providing focused research highlights

    Target: ≤40% of context window for efficient token usage.
    """
    key_insights: List[str]  # Max 10 insights from all previous sections
    previous_sections: List[str]  # Compressed summaries of recent sections
    research_highlights: str  # Relevant excerpts from Q&A history
    total_tokens: int  # Estimated token count for this context

    def is_within_budget(self, budget_tokens: int = 8000) -> bool:
        """Check if context fits within token budget (default: 8K of 20K window)."""
        return self.total_tokens <= budget_tokens


@dataclass
class ValidationResult:
    """
    Quality validation results for a generated section.

    Tracks compliance with quality metrics:
    - Depth: ≥300 words
    - Citations: ≥1 per 150 words
    - Redundancy: <70% similarity with previous sections
    - Coherence: ≥0.8 semantic similarity at transitions
    """
    is_valid: bool
    section_id: str
    issues: List[str] = field(default_factory=list)
    depth_score: Optional[float] = None  # Word count / target
    citation_score: Optional[float] = None  # Citations per 150 words
    redundancy_score: Optional[float] = None  # Max similarity with previous sections
    coherence_score: Optional[float] = None  # Transition quality

    def should_regenerate(self, attempt: int, max_attempts: int = 2) -> bool:
        """Determine if section should be regenerated based on validation results."""
        if attempt >= max_attempts:
            return False  # Exceeded max attempts
        return not self.is_valid

    def get_regeneration_guidance(self) -> str:
        """Generate specific guidance for regeneration based on identified issues."""
        if not self.issues:
            return ""

        guidance_parts = ["Address the following issues in regeneration:"]
        guidance_parts.extend(f"- {issue}" for issue in self.issues)
        return "\n".join(guidance_parts)
