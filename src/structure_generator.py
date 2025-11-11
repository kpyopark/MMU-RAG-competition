"""
Report structure generation component.

This module analyzes user queries and research context to generate comprehensive
report structures with multi-perspective chapter organization. It determines:
- Relevant analytical perspectives (Financial, Technical, Regulatory, etc.)
- Chapter organization (4-6 chapters)
- Section breakdown within chapters (3-5 sections each)
- Target word counts and generation guidance
"""

import json
from typing import List
from loguru import logger

from .report_structure import ReportStructure, Chapter, SectionSpec
from .generator import get_llm_response


# Standard analytical perspectives for multi-faceted analysis
STANDARD_PERSPECTIVES = [
    "Financial/Economic",
    "Technical/Operational",
    "Regulatory/Legal",
    "Strategic/Competitive",
    "Risk/Challenge",
    "Market/Industry",
]


STRUCTURE_GENERATION_PROMPT = """You are a research report structuring expert. Your task is to analyze a user query and create a comprehensive report structure with multiple analytical perspectives.

**User Query:**
{query}

**Research Plan:**
{plan}

**Research Summary:**
{research_summary}

Based on the query complexity and research scope, create a structured report outline that:

1. **Executive Summary**: High-level synthesis (1 section, ~400 words)

2. **Main Chapters** (4-6 chapters):
   - Each chapter should address ONE major analytical perspective
   - Relevant perspectives: Financial/Economic, Technical/Operational, Regulatory/Legal, Strategic/Competitive, Risk/Challenge, Market/Industry
   - Choose 4-6 most relevant perspectives based on query focus

3. **Chapter Sections** (3-5 sections per chapter):
   - Each section should drill into a specific aspect within the chapter's perspective
   - Target: 300-500 words per section for detailed analysis
   - Provide clear guidance on what each section should cover

4. **Conclusion**: Forward-looking synthesis and implications (1 section, ~400 words)

**Guidelines:**
- Simple queries (single aspect): 2-3 chapters
- Moderate queries (2-3 aspects): 4-5 chapters
- Complex queries (4+ aspects): 5-7 chapters
- Each section must add unique value (no redundancy)
- Sections should build logically within chapters
- Total report target: 2,500-4,000 words

**Output Format (JSON):**
{{
  "executive_summary": {{
    "title": "Executive Summary",
    "guidance": "High-level synthesis covering all key perspectives and findings"
  }},
  "chapters": [
    {{
      "title": "Chapter Title",
      "perspective": "Primary Perspective (e.g., Financial/Economic)",
      "sections": [
        {{
          "title": "Section Title",
          "guidance": "Specific focus and key points to cover",
          "target_word_count": 350
        }}
      ]
    }}
  ],
  "conclusion": {{
    "title": "Conclusion and Implications",
    "guidance": "Forward-looking synthesis, recommendations, future outlook"
  }}
}}

Generate the report structure now."""


PERSPECTIVE_ANALYSIS_PROMPT = """Analyze the following user query and identify the most relevant analytical perspectives for a comprehensive research report.

**User Query:**
{query}

**Available Perspectives:**
- Financial/Economic: Deal structure, valuation, revenue impact, financial metrics
- Technical/Operational: Technology, implementation, operational details, capabilities
- Regulatory/Legal: Compliance, legal issues, regulatory approval, antitrust
- Strategic/Competitive: Market positioning, competitive dynamics, strategic rationale
- Risk/Challenge: Implementation risks, market risks, execution challenges
- Market/Industry: Industry trends, market landscape, broader implications

**Instructions:**
1. Identify 4-6 most relevant perspectives based on query focus
2. Rank them by importance to answering the query
3. Explain why each perspective is relevant

**Output Format (JSON):**
{{
  "perspectives": [
    {{
      "name": "Perspective Name",
      "relevance_score": 9,
      "rationale": "Why this perspective is important for this query"
    }}
  ]
}}

Generate the perspective analysis now."""


class ReportStructureGenerator:
    """
    Generates comprehensive report structures with multi-perspective analysis.

    This component analyzes user queries and research context to determine:
    - Which analytical perspectives are most relevant
    - How to organize chapters by perspective
    - How to break down chapters into focused sections
    - Appropriate guidance for each section's content
    """

    def __init__(self):
        """Initialize the structure generator."""
        self.standard_perspectives = STANDARD_PERSPECTIVES

    def analyze_query_perspectives(self, query: str) -> List[dict]:
        """
        Analyze query to identify relevant analytical perspectives.

        Args:
            query: User's research query

        Returns:
            List of perspective dicts with name, relevance_score, rationale
            Sorted by relevance_score (descending)
        """
        logger.info(f"Analyzing query for relevant perspectives: {query[:100]}...")

        prompt = PERSPECTIVE_ANALYSIS_PROMPT.format(query=query)

        try:
            response = get_llm_response(
                prompt=prompt,
                system_prompt="You are a research planning expert. Output valid JSON only.",
            )

            # Parse JSON response
            response_clean = self._extract_json(response)
            analysis = json.loads(response_clean)

            perspectives = analysis.get("perspectives", [])

            # Sort by relevance score
            perspectives.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)

            logger.info(f"Identified {len(perspectives)} relevant perspectives")
            for p in perspectives[:3]:  # Log top 3
                logger.debug(
                    f"  - {p['name']} (score: {p['relevance_score']}): {p['rationale'][:80]}..."
                )

            return perspectives

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse perspective analysis: {e}, using defaults")
            # Fallback: return all standard perspectives with equal weight
            return [
                {
                    "name": p,
                    "relevance_score": 5,
                    "rationale": "Default perspective for comprehensive analysis",
                }
                for p in self.standard_perspectives[:4]
            ]

    def generate_chapter_outline(
        self, query: str, plan: str, research_summary: str
    ) -> ReportStructure:
        """
        Generate complete report structure with chapters and sections.

        Args:
            query: User's research query
            plan: Research plan from planning phase
            research_summary: Summary of research findings from Q&A history

        Returns:
            ReportStructure with executive summary, chapters, conclusion
        """
        logger.info("Generating comprehensive report structure...")

        prompt = STRUCTURE_GENERATION_PROMPT.format(
            query=query, plan=plan, research_summary=research_summary
        )

        try:
            response = get_llm_response(
                prompt=prompt,
                system_prompt="You are a research report structuring expert. Output valid JSON only.",
            )

            # Extract and parse JSON
            response_clean = self._extract_json(response)
            structure_data = json.loads(response_clean)

            # Build ReportStructure from JSON
            report_structure = self._build_report_structure(structure_data)

            logger.info(
                f"Generated report structure: {len(report_structure.chapters)} chapters, "
                f"{report_structure.total_sections()} total sections, "
                f"~{report_structure.estimated_word_count} words"
            )

            return report_structure

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse structure generation: {e}")
            logger.warning("Falling back to default report structure")
            return self._generate_default_structure(query)

    def _build_report_structure(self, structure_data: dict) -> ReportStructure:
        """
        Build ReportStructure object from JSON data.

        Args:
            structure_data: Parsed JSON structure from LLM

        Returns:
            ReportStructure instance
        """
        # Build executive summary
        exec_data = structure_data["executive_summary"]
        executive_summary = SectionSpec(
            title=exec_data["title"],
            chapter_number=0,
            section_number=1,
            perspective="Executive Summary",
            guidance=exec_data["guidance"],
            target_word_count=400,
        )

        # Build chapters
        chapters = []
        for ch_idx, ch_data in enumerate(structure_data["chapters"], start=1):
            sections = []
            for sec_idx, sec_data in enumerate(ch_data["sections"], start=1):
                section = SectionSpec(
                    title=sec_data["title"],
                    chapter_number=ch_idx,
                    section_number=sec_idx,
                    perspective=ch_data["perspective"],
                    guidance=sec_data["guidance"],
                    target_word_count=sec_data.get("target_word_count", 350),
                )
                sections.append(section)

            chapter = Chapter(
                title=ch_data["title"],
                perspective=ch_data["perspective"],
                sections=sections,
                chapter_number=ch_idx,
            )
            chapters.append(chapter)

        # Build conclusion
        concl_data = structure_data["conclusion"]
        conclusion = SectionSpec(
            title=concl_data["title"],
            chapter_number=len(chapters) + 1,
            section_number=1,
            perspective="Conclusion",
            guidance=concl_data["guidance"],
            target_word_count=400,
        )

        # Calculate estimates
        estimated_sections = 2 + sum(len(ch.sections) for ch in chapters)
        estimated_word_count = (
            400  # Executive summary
            + sum(ch.total_target_words() for ch in chapters)
            + 400  # Conclusion
        )

        return ReportStructure(
            executive_summary=executive_summary,
            chapters=chapters,
            conclusion=conclusion,
            estimated_word_count=estimated_word_count,
            estimated_sections=estimated_sections,
        )

    def _generate_default_structure(self, query: str) -> ReportStructure:
        """
        Generate fallback default report structure if LLM parsing fails.

        Args:
            query: User's research query

        Returns:
            ReportStructure with basic 3-chapter structure
        """
        logger.warning("Using default 3-chapter report structure")

        executive_summary = SectionSpec(
            title="Executive Summary",
            chapter_number=0,
            section_number=1,
            perspective="Executive Summary",
            guidance="Provide high-level synthesis of key findings",
            target_word_count=400,
        )

        # Default chapters with standard sections
        chapters = [
            Chapter(
                title="Background and Context",
                perspective="General Analysis",
                sections=[
                    SectionSpec(
                        title="Overview",
                        chapter_number=1,
                        section_number=1,
                        perspective="General Analysis",
                        guidance="Provide context and background",
                        target_word_count=350,
                    ),
                    SectionSpec(
                        title="Key Details",
                        chapter_number=1,
                        section_number=2,
                        perspective="General Analysis",
                        guidance="Present essential facts and details",
                        target_word_count=350,
                    ),
                ],
                chapter_number=1,
            ),
            Chapter(
                title="Analysis and Implications",
                perspective="Strategic Analysis",
                sections=[
                    SectionSpec(
                        title="Primary Analysis",
                        chapter_number=2,
                        section_number=1,
                        perspective="Strategic Analysis",
                        guidance="Analyze main implications",
                        target_word_count=350,
                    ),
                    SectionSpec(
                        title="Secondary Considerations",
                        chapter_number=2,
                        section_number=2,
                        perspective="Strategic Analysis",
                        guidance="Explore additional factors",
                        target_word_count=350,
                    ),
                ],
                chapter_number=2,
            ),
            Chapter(
                title="Future Outlook",
                perspective="Forward-Looking",
                sections=[
                    SectionSpec(
                        title="Expected Developments",
                        chapter_number=3,
                        section_number=1,
                        perspective="Forward-Looking",
                        guidance="Discuss future trajectories",
                        target_word_count=350,
                    ),
                    SectionSpec(
                        title="Potential Scenarios",
                        chapter_number=3,
                        section_number=2,
                        perspective="Forward-Looking",
                        guidance="Consider alternative outcomes",
                        target_word_count=350,
                    ),
                ],
                chapter_number=3,
            ),
        ]

        conclusion = SectionSpec(
            title="Conclusion",
            chapter_number=4,
            section_number=1,
            perspective="Conclusion",
            guidance="Synthesize findings and provide recommendations",
            target_word_count=400,
        )

        estimated_sections = 2 + sum(len(ch.sections) for ch in chapters)
        estimated_word_count = 400 + sum(ch.total_target_words() for ch in chapters) + 400

        return ReportStructure(
            executive_summary=executive_summary,
            chapters=chapters,
            conclusion=conclusion,
            estimated_word_count=estimated_word_count,
            estimated_sections=estimated_sections,
        )

    def _extract_json(self, response: str) -> str:
        """
        Extract JSON from LLM response (handles markdown code blocks).

        Args:
            response: Raw LLM response potentially containing JSON in code blocks

        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        return response.strip()
