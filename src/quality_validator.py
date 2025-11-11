"""
Quality validation component for generated sections.

This module validates generated sections against quality metrics:
- Depth: ≥300 words target
- Citations: ≥1 per 150 words
- Redundancy: <70% similarity with previous sections
- Coherence: Logical flow and transitions

Triggers regeneration when quality thresholds are not met.
"""

from typing import List
from loguru import logger

from .report_structure import GeneratedSection, ValidationResult


# Quality thresholds
MIN_WORD_COUNT = 300
TARGET_WORD_COUNT = 350
MIN_CITATION_DENSITY = 1.0 / 150  # 1 citation per 150 words
MAX_REDUNDANCY_SIMILARITY = 0.70  # 70% similarity threshold
MIN_COHERENCE_SCORE = 0.8


class QualityValidator:
    """
    Validates section quality and determines regeneration needs.

    Quality Metrics:
    - Depth: Word count ≥300 (target: 350)
    - Citations: ≥1 per 150 words
    - Redundancy: <70% similarity with previous sections
    - Coherence: Smooth transitions and logical flow

    Regeneration Logic:
    - First attempt: Allow minor quality issues
    - Second attempt: Strict quality enforcement
    - Third attempt: Accept with warnings
    """

    def __init__(self):
        """Initialize quality validator."""
        logger.info("Initialized QualityValidator")

    def validate_section(
        self,
        section: GeneratedSection,
        previous_sections: List[GeneratedSection],
        attempt: int = 1,
    ) -> ValidationResult:
        """
        Validate section quality against all metrics.

        Args:
            section: Generated section to validate
            previous_sections: All previously generated sections
            attempt: Generation attempt number (1-based)

        Returns:
            ValidationResult with validation status and scores
        """
        logger.debug(
            f"Validating section {section.get_section_id()} (attempt {attempt})"
        )

        issues = []

        # 1. Depth check: word count
        depth_score = section.word_count / TARGET_WORD_COUNT
        if section.word_count < MIN_WORD_COUNT:
            issues.append(
                f"Insufficient depth: {section.word_count} words (minimum: {MIN_WORD_COUNT})"
            )
            logger.warning(
                f"Section {section.get_section_id()} failed depth check: "
                f"{section.word_count} < {MIN_WORD_COUNT} words"
            )

        # 2. Citation density check
        citation_score = section.citation_density()
        if citation_score < MIN_CITATION_DENSITY:
            issues.append(
                f"Insufficient citations: {len(section.citations_used)} citations "
                f"for {section.word_count} words (target: ≥{MIN_CITATION_DENSITY * section.word_count:.1f})"
            )
            logger.warning(
                f"Section {section.get_section_id()} failed citation check: "
                f"density {citation_score:.3f} < {MIN_CITATION_DENSITY:.3f}"
            )

        # 3. Redundancy check (simplified: text overlap with previous sections)
        redundancy_score = 0.0
        if previous_sections:
            redundancy_score = self._check_redundancy(section, previous_sections)
            if redundancy_score > MAX_REDUNDANCY_SIMILARITY:
                issues.append(
                    f"High redundancy: {redundancy_score * 100:.0f}% similarity with previous sections "
                    f"(threshold: {MAX_REDUNDANCY_SIMILARITY * 100:.0f}%)"
                )
                logger.warning(
                    f"Section {section.get_section_id()} failed redundancy check: "
                    f"{redundancy_score:.2f} > {MAX_REDUNDANCY_SIMILARITY:.2f}"
                )

        # 4. Coherence check (basic: check if section is not placeholder/error)
        coherence_score = self._check_coherence(section)
        if coherence_score < MIN_COHERENCE_SCORE:
            issues.append(
                f"Poor coherence: Section appears to be placeholder or error content"
            )
            logger.warning(
                f"Section {section.get_section_id()} failed coherence check"
            )

        # Determine validation result
        is_valid = len(issues) == 0

        result = ValidationResult(
            is_valid=is_valid,
            section_id=section.get_section_id(),
            issues=issues,
            depth_score=depth_score,
            citation_score=citation_score,
            redundancy_score=redundancy_score,
            coherence_score=coherence_score,
        )

        if is_valid:
            logger.info(
                f"Section {section.get_section_id()} passed validation "
                f"(depth: {depth_score:.2f}, citations: {citation_score:.3f}, "
                f"redundancy: {redundancy_score:.2f}, coherence: {coherence_score:.2f})"
            )
        else:
            logger.warning(
                f"Section {section.get_section_id()} failed validation with {len(issues)} issues"
            )

        return result

    def should_regenerate(
        self, validation_result: ValidationResult, attempt: int, max_attempts: int = 2
    ) -> tuple[bool, str]:
        """
        Determine if section should be regenerated.

        Args:
            validation_result: Validation result from validate_section()
            attempt: Current generation attempt (1-based)
            max_attempts: Maximum regeneration attempts

        Returns:
            Tuple of (should_regenerate, regeneration_guidance)
        """
        if attempt >= max_attempts:
            logger.info(
                f"Section {validation_result.section_id}: "
                f"Max attempts ({max_attempts}) reached, accepting section"
            )
            return False, ""

        if validation_result.is_valid:
            return False, ""

        # Build regeneration guidance
        guidance = validation_result.get_regeneration_guidance()

        logger.info(
            f"Section {validation_result.section_id}: "
            f"Regeneration needed (attempt {attempt}/{max_attempts})"
        )

        return True, guidance

    def _check_redundancy(
        self, section: GeneratedSection, previous_sections: List[GeneratedSection]
    ) -> float:
        """
        Check redundancy with previous sections (simplified word overlap).

        Args:
            section: Current section
            previous_sections: All previous sections

        Returns:
            Redundancy score (0.0 to 1.0, higher = more redundant)
        """
        # Simplified redundancy: word overlap with most similar section
        current_words = set(section.content.lower().split())

        max_overlap = 0.0
        for prev_section in previous_sections:
            prev_words = set(prev_section.content.lower().split())

            # Calculate Jaccard similarity
            intersection = len(current_words & prev_words)
            union = len(current_words | prev_words)

            if union > 0:
                similarity = intersection / union
                max_overlap = max(max_overlap, similarity)

        return max_overlap

    def _check_coherence(self, section: GeneratedSection) -> float:
        """
        Check section coherence (simplified: detect placeholder/error content).

        Args:
            section: Generated section

        Returns:
            Coherence score (0.0 to 1.0, higher = more coherent)
        """
        content_lower = section.content.lower()

        # Check for error indicators
        error_indicators = [
            "generation failed",
            "error:",
            "[content generation failed",
            "not implemented",
            "placeholder",
        ]

        for indicator in error_indicators:
            if indicator in content_lower:
                return 0.0  # Failed coherence

        # Check for minimum structure (has paragraphs, sentences)
        has_paragraphs = "\n\n" in section.content or "\n" in section.content
        has_sentences = ". " in section.content or ".\n" in section.content

        if not (has_paragraphs and has_sentences):
            return 0.5  # Marginal coherence

        return 1.0  # Good coherence
