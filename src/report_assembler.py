"""
Report assembly component for final markdown generation.

This module assembles individual sections into a cohesive multi-chapter report with:
- Markdown H1/H2/H3 hierarchy
- Chapter organization
- Inline citations
- Organized citations section (grouped by chapter)
- Report metadata (word count, generation time, section count)
"""

from typing import List
from collections import defaultdict
from loguru import logger

from .report_structure import ReportStructure, GeneratedSection


class ReportAssembler:
    """
    Assembles generated sections into final markdown report.

    Output Format:
    - Executive Summary (H1)
    - Chapters (H1) with Sections (H2)
    - Conclusion (H1)
    - Citations (H1) organized by chapter
    - Report Metadata footer
    """

    def __init__(self):
        """Initialize report assembler."""
        logger.info("Initialized ReportAssembler")

    def assemble_final_report(
        self, structure: ReportStructure, sections: List[GeneratedSection]
    ) -> str:
        """
        Assemble all sections into final markdown report.

        Args:
            structure: Report structure with chapter organization
            sections: All generated sections in order

        Returns:
            Complete markdown report string
        """
        logger.info(
            f"Assembling final report: {len(sections)} sections, "
            f"{len(structure.chapters)} chapters"
        )

        parts = []

        # Build section lookup by spec ID
        section_map = {s.spec.get_full_id(): s for s in sections}

        # 1. Executive Summary
        exec_section = section_map.get("0.1")
        if exec_section:
            parts.append("# Executive Summary\n")
            parts.append(exec_section.content)
            parts.append("\n\n---\n")

        # 2. Main Chapters
        for chapter in structure.chapters:
            parts.append(f"\n# Chapter {chapter.chapter_number}: {chapter.title}\n")
            parts.append(f"*Perspective: {chapter.perspective}*\n")

            for section_spec in chapter.sections:
                section = section_map.get(section_spec.get_full_id())
                if section:
                    parts.append(f"\n## {section.spec.get_full_id()} {section.spec.title}\n")
                    parts.append(section.content)
                    parts.append("\n")

            parts.append("\n---\n")

        # 3. Conclusion
        concl_id = structure.conclusion.get_full_id()
        concl_section = section_map.get(concl_id)
        if concl_section:
            parts.append("\n# Conclusion\n")
            parts.append(concl_section.content)
            parts.append("\n\n---\n")

        # 4. Citations (organized by chapter)
        citations_section = self.organize_citations_by_chapter(structure, sections)
        parts.append(citations_section)

        # 5. Report Metadata
        metadata = self.generate_metadata(structure, sections)
        parts.append(metadata)

        final_report = "".join(parts)

        logger.info(
            f"Final report assembled: {len(final_report)} characters, "
            f"{len(final_report.split())} words"
        )

        return final_report

    def organize_citations_by_chapter(
        self, structure: ReportStructure, sections: List[GeneratedSection]
    ) -> str:
        """
        Organize citations by chapter for easier reference.

        Args:
            structure: Report structure with chapters
            sections: Generated sections with citations

        Returns:
            Markdown string with citations organized by chapter
        """
        logger.debug("Organizing citations by chapter")

        # Collect all unique citations across all sections
        all_citations = []
        for section in sections:
            all_citations.extend(section.citations_used)

        # Deduplicate while preserving order
        unique_citations = []
        seen = set()
        for citation in all_citations:
            if citation not in seen:
                unique_citations.append(citation)
                seen.add(citation)

        if not unique_citations:
            return "\n# Citations\n\nNo citations available for this report.\n"

        parts = ["\n# Citations\n"]

        # Group citations by chapter
        citations_by_chapter = defaultdict(list)

        # Map sections to chapters
        for section in sections:
            chapter_num = section.spec.chapter_number
            if section.citations_used:
                citations_by_chapter[chapter_num].extend(section.citations_used)

        # Format citations by chapter
        for chapter_num in sorted(citations_by_chapter.keys()):
            if chapter_num == 0:
                parts.append("\n## Executive Summary\n")
            elif chapter_num == len(structure.chapters) + 1:
                parts.append("\n## Conclusion\n")
            else:
                chapter = structure.chapters[chapter_num - 1]
                parts.append(f"\n## Chapter {chapter_num}: {chapter.title}\n")

            # Deduplicate citations for this chapter
            chapter_citations = []
            seen_in_chapter = set()
            for cit in citations_by_chapter[chapter_num]:
                if cit not in seen_in_chapter:
                    chapter_citations.append(cit)
                    seen_in_chapter.add(cit)

            for citation in chapter_citations:
                parts.append(f"- [{citation}]\n")

        parts.append("\n")

        return "".join(parts)

    def generate_metadata(
        self, structure: ReportStructure, sections: List[GeneratedSection]
    ) -> str:
        """
        Generate report metadata footer.

        Args:
            structure: Report structure
            sections: Generated sections

        Returns:
            Markdown string with metadata
        """
        total_words = sum(s.word_count for s in sections)
        total_sections = len(sections)
        total_citations = sum(len(s.citations_used) for s in sections)
        total_time = sum(s.generation_time for s in sections)

        avg_words_per_section = total_words / total_sections if total_sections > 0 else 0
        citation_density = (total_citations / total_words) * 150 if total_words > 0 else 0

        metadata = f"""

---

## Report Metadata

**Generated Report Statistics:**
- **Total Word Count:** {total_words:,} words
- **Total Sections:** {total_sections} sections ({len(structure.chapters)} chapters)
- **Total Citations:** {total_citations} sources
- **Average Section Length:** {avg_words_per_section:.0f} words
- **Citation Density:** {citation_density:.2f} citations per 150 words
- **Total Generation Time:** {total_time:.1f} seconds ({total_time / 60:.1f} minutes)

**Report Structure:**
- Executive Summary: 1 section
- Main Chapters: {len(structure.chapters)} chapters
- Conclusion: 1 section

*Generated by TTD-DR Structured Report Generation System*
"""

        return metadata
