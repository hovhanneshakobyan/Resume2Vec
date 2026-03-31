

from __future__ import annotations

from typing import Dict, List

from src.scorer import ATSScorer


class ResumeOptimizer:
    """
    Creates a safer ATS-optimized draft by:
    - adding a targeted summary
    - adding a target-role keyword block
    - preserving original content
    - not inventing fake achievements
    """

    def __init__(self):
        self.scorer = ATSScorer()

    def optimize(self, resume_text: str, jd_text: str) -> Dict:
        score = self.scorer.score(resume_text, jd_text)
        keyword_report = score["keyword_report"]

        mandatory_missing = keyword_report.get("mandatory_missing", [])
        optional_missing = keyword_report.get("optional_missing", [])
        matched = keyword_report.get("matched", [])

        optimized = resume_text.strip()

        additions: List[str] = []
        suggestions: List[str] = []

        target_summary = self._build_target_summary(jd_text, matched, mandatory_missing)
        if "PROFESSIONAL SUMMARY" not in optimized.upper() and "SUMMARY" not in optimized.upper():
            additions.append(
                "PROFESSIONAL SUMMARY\n"
                f"{target_summary}\n"
            )

        target_skills_block = self._build_target_skills_block(matched, mandatory_missing, optional_missing)
        additions.append(target_skills_block)

        if mandatory_missing:
            suggestions.append(
                "Highest-impact missing requirements: " + ", ".join(mandatory_missing[:6])
            )

        if optional_missing:
            suggestions.append(
                "Nice-to-have missing requirements: " + ", ".join(optional_missing[:6])
            )

        if "No clear experience bullet points detected; ATS evidence for responsibilities may be weak." in score["format_warnings"]:
            suggestions.append(
                "Add 2–4 bullets under your most relevant role. Example themes: backend/API work, testing/debugging, UI work, reporting, collaboration."
            )

        if "Likely visually designed header/contact layout; ATS readability may be reduced." in score["format_warnings"]:
            suggestions.append(
                "Prepare a one-column ATS version without icons, photo, or multi-column layout."
            )

        if additions:
            optimized = "\n\n".join(additions) + "\n\n" + optimized

        return {
            "optimized_resume": optimized,
            "changed_bullets": [],
            "suggestions": suggestions,
        }

    def _build_target_summary(self, jd_text: str, matched: List[str], mandatory_missing: List[str]) -> str:
        jd_low = jd_text.lower()

        role_hint = "software developer"
        if "junior" in jd_low and "developer" in jd_low:
            role_hint = "junior developer"
        elif "specialist" in jd_low:
            role_hint = "specialist"
        elif "engineer" in jd_low:
            role_hint = "engineer"

        strong = [m.replace(" (related)", "") for m in matched[:5]]
        if strong:
            skill_text = ", ".join(strong[:4])
        else:
            skill_text = "software development"

        if mandatory_missing:
            return (
                f"Candidate targeting {role_hint} roles with relevant background in {skill_text}. "
                f"Resume should be strengthened with clearer evidence for: {', '.join(mandatory_missing[:4])}."
            )

        return (
            f"Candidate targeting {role_hint} roles with relevant background in {skill_text}. "
            f"Resume includes partially aligned technical experience and should emphasize impact, responsibilities, and ATS-readable structure."
        )

    def _build_target_skills_block(self, matched: List[str], mandatory_missing: List[str], optional_missing: List[str]) -> str:
        lines = ["TARGET ROLE ALIGNMENT"]

        if matched:
            lines.append("Aligned skills: " + ", ".join(matched[:10]))

        if mandatory_missing:
            lines.append("Missing mandatory evidence: " + ", ".join(mandatory_missing[:8]))

        if optional_missing:
            lines.append("Missing optional evidence: " + ", ".join(optional_missing[:8]))

        return "\n".join(lines)