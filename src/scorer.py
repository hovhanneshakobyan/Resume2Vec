from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from src.jd_rules import build_requirement_table
from src.sections import SectionExtractor
from src.semantic_model import SemanticResumeMatcher
from src.utils import ACTION_VERBS, count_numeric_impact, looks_like_title_only


SKILL_PATTERNS = {
    "c#": [r"\bc#\b", r"\bc sharp\b"],
    ".net": [r"\.net\b", r"\bdotnet\b"],
    "asp.net": [r"\basp\.net\b", r"\basp net\b"],
    "sql": [r"\bsql\b"],
    "sql server": [r"\bsql server\b"],
    "postgresql": [r"\bpostgresql\b", r"\bpostgres\b"],
    "javascript": [r"\bjavascript\b", r"\bjavasript\b", r"\bjs\b"],
    "html5": [r"\bhtml5\b"],
    "css3": [r"\bcss3\b"],
    "wpf": [r"\bwpf\b"],
    "maui": [r"\bmaui\b"],
    "android": [r"\bandroid\b"],
    "azure devops": [r"\bazure devops\b"],
    "rest api": [r"\brest api\b", r"\brestful api\b"],
    "entity framework": [r"\bentity framework\b", r"\bef core\b"],
    "unit testing": [r"\bunit testing\b", r"\bautomated testing\b"],
    "oop": [r"\boop\b", r"\bobject-oriented\b", r"\bobject oriented\b"],
    "design patterns": [r"\bdesign patterns\b", r"\bprogramming patterns\b"],
    "full stack": [r"\bfull stack\b", r"\bfull-stack\b"],
    "web ui": [r"\bweb ui\b", r"\bweb applications\b", r"\bfrontend\b"],
    "desktop ui": [r"\bdesktop ui\b", r"\bdesktop applications\b"],
    "python": [r"\bpython\b"],
    "java": [r"\bjava\b"],
    "c++": [r"\bc\+\+\b", r"\bcpp\b", r"\bc\+\+ builder\b"],
    "kotlin": [r"\bkotlin\b"],
    "delphi": [r"\bdelphi\b"],
    "vcl": [r"\bvcl\b", r"\bvisual component library\b"],
    "rad studio": [r"\brad studio\b", r"\bembarcadero rad studio\b"],
    "devexpress": [r"\bdevexpress\b", r"\bdevexpress vcl\b"],
    "winapi": [r"\bwinapi\b", r"\bwindows api\b"],
    "solar": [r"\bsolar\b", r"\bsmart pv\b", r"\bpv\b"],
    "inverters": [r"\binverters\b", r"\bsolar inverters\b"],
    "bms": [r"\bbms\b", r"\bbattery management systems\b"],
    "ems": [r"\bems\b", r"\benergy management systems\b"],
    "storage": [r"\bstorage\b", r"\bess\b"],
    "sales": [r"\bsales\b", r"\bsales management\b"],
    "business development": [r"\bbusiness development\b"],
    "electrical engineering": [r"\belectrical engineering\b"],
    "power electronics": [r"\bpower electronics\b"],
    "english": [r"\benglish\b", r"\bupper-intermediate\b", r"\bfluent\b"],
    "armenian": [r"\barmenian\b"],
}


RELATED_SKILLS = {
    "android": {"maui", "kotlin"},
    "maui": {"android", ".net"},
    "asp.net": {".net", "c#", "rest api", "sql"},
    "oop": {"c#", "c++", "java", "design patterns", "delphi"},
    "design patterns": {"oop", "c#", "c++", "java", "delphi"},
    "full stack": {"asp.net", "javascript", "sql", "html5", "css3"},
    "web ui": {"asp.net", "javascript", "html5", "css3"},
    "desktop ui": {"wpf", "maui", "vcl", "winapi"},
    "vcl": {"delphi", "winapi", "desktop ui"},
    "solar": {"inverters", "storage", "bms", "ems", "electrical engineering", "power electronics"},
    "storage": {"bms", "ems", "solar", "inverters"},
    "sales": {"business development"},
    "business development": {"sales"},
}


class ATSScorer:
    def __init__(self):
        self.sections = SectionExtractor()
        self.semantic = SemanticResumeMatcher()

    def extract_skills(self, text: str) -> List[str]:
        low = text.lower()
        found = []

        for skill, patterns in SKILL_PATTERNS.items():
            if any(re.search(p, low) for p in patterns):
                found.append(skill)

        if "javascript" in found and "java" in found:
            found.remove("java")

        return sorted(set(found))

    def extract_experience_bullets(self, exp_block: str) -> List[str]:
        lines = [ln.strip() for ln in exp_block.splitlines() if ln.strip()]
        bullets = []

        for line in lines:
            low = line.lower()

            if re.fullmatch(r"[A-Z ,&/()\-\+]+", line):
                continue

            if any(word in low for word in ["scholarship", "award", "publication", "journal", "resident", "finalist"]):
                continue

            if re.match(r"^[•\-\*]", line):
                clean = re.sub(r"^[•\-\*]\s*", "", line).strip()
                if len(clean.split()) >= 5:
                    bullets.append(clean)
                continue

            if len(line.split()) >= 8 and not looks_like_title_only(line):
                bullets.append(line)

        return bullets

    def detect_format_issues(self, resume_text: str, exp_bullets: List[str]) -> List[str]:
        warnings = []
        lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
        short_lines = [ln for ln in lines if len(ln.split()) <= 4]

        if len(lines) > 20 and len(short_lines) / max(len(lines), 1) > 0.45:
            warnings.append("Resume appears heavily fragmented into short visual lines; parsing order may be unreliable.")

        if "@" in resume_text and "github" in resume_text.lower() and len(short_lines) > 10:
            warnings.append("Likely visually designed header/contact layout; ATS readability may be reduced.")

        if not exp_bullets:
            warnings.append("No clear experience bullet points detected; ATS evidence for responsibilities may be weak.")

        return warnings

    def extract_experience_titles(selfself,exp_block:str) -> List[str]:
        '''
        extract job titles / roles line from expereice nsection
        these are weaker than bullets but still something useful for semantic matching
        '''
        lines = [ln.strip() for ln in exp_block.splitlines() if ln.strip()]
        titles = []
        for line in lines:
            low = line.lower()

            if re.fullmatch(r"[A-Z ,&/()\-\+]+", line):
                continue
            if any(word in low for word in ["scholarship" , "award", "publication", "journal","resident" , "finalist"]):
                continue
            if looks_like_title_only(line) and 2 <= len(line.split()) <= 10:
                titles.append(line)
        return titles

    def _related_match_score(self, required_skill: str, resume_skills: set[str]) -> float:
        if required_skill in resume_skills:
            return 1.0

        related = RELATED_SKILLS.get(required_skill, set())
        if any(skill in resume_skills for skill in related):
            return 0.45

        return 0.0

    def keyword_report(self, resume_text: str, jd_text: str) -> Dict:
        resume_skills = set(self.extract_skills(resume_text))
        requirement_rows = build_requirement_table(jd_text, self.extract_skills)

        matched = []
        missing = []
        optional_missing = []
        mandatory_missing = []

        earned = 0.0
        total = 0.0

        for row in requirement_rows:
            priority = row["priority"]
            skills = row["skills"]

            if not skills:
                continue

            if priority == "mandatory":
                weight = 1.0
            elif priority == "optional":
                weight = 0.35
            else:
                weight = 0.6

            for skill in skills:
                total += weight
                score = self._related_match_score(skill, resume_skills)

                if score >= 1.0:
                    matched.append(skill)
                    earned += weight
                elif score > 0:
                    matched.append(f"{skill} (related)")
                    earned += weight * score
                else:
                    missing.append(skill)
                    if priority == "mandatory":
                        mandatory_missing.append(skill)
                    elif priority == "optional":
                        optional_missing.append(skill)

        coverage = (earned / max(total, 1e-9)) * 100.0

        return {
            "matched": sorted(set(matched)),
            "missing": sorted(set(missing)),
            "mandatory_missing": sorted(set(mandatory_missing)),
            "optional_missing": sorted(set(optional_missing)),
            "coverage_pct": round(coverage, 1),
        }

    def semantic_score(
            self,
            resume_text: str,
            jd_text: str,
            exp_bullets: List[str],
            exp_titles: List[str],
            resume_education: List[str],
    ) -> Dict:
        jd_rows = build_requirement_table(jd_text, self.extract_skills)

        jd_resp_lines = []
        jd_skill_lines = []
        jd_edu_lines = []

        for row in jd_rows:
            low = row["text"].lower()

            if any(x in low for x in
                   ["bachelor", "master", "degree", "computer science", "electrical engineering", "power electronics"]):
                jd_edu_lines.append(row["text"])
            elif row["skills"]:
                jd_skill_lines.append(row["text"])
            else:
                jd_resp_lines.append(row["text"])

        def score_group(left: List[str], right: List[str]) -> float:
            if not left or not right:
                return 0.0
            left_emb = self.semantic.encode(left)
            right_emb = self.semantic.encode(right)
            sim = np.matmul(left_emb, right_emb.T)
            return float(sim.max(axis=1).mean())

        # -----------------------------
        # Responsibility scoring logic
        # -----------------------------
        bullet_score = score_group(jd_resp_lines, exp_bullets)

        title_score = 0.0
        if exp_titles:
            title_score = score_group(jd_resp_lines, exp_titles)

        skill_hint_score = 0.0
        resume_skills = self.extract_skills(resume_text)
        if resume_skills:
            skill_hint_score = score_group(jd_resp_lines, resume_skills)

        if exp_bullets:
            # Strongest evidence: real bullets
            responsibility_score = bullet_score
        elif exp_titles:
            # Weak fallback: titles help, but cap them
            responsibility_score = min(0.75 * title_score + 0.25 * skill_hint_score, 0.35)
        elif resume_skills:
            # Very weak fallback: only skill hints
            responsibility_score = min(skill_hint_score, 0.15)
        else:
            responsibility_score = 0.0

        # -----------------------------
        # Other semantic components
        # -----------------------------
        skill_score = score_group(jd_skill_lines, resume_skills)
        education_score = score_group(jd_edu_lines, resume_education)

        overall = (
                0.45 * responsibility_score +
                0.35 * skill_score +
                0.20 * education_score
        )

        return {
            "overall_similarity": round(overall, 4),
            "responsibility_score": round(responsibility_score, 4),
            "responsibility_bullet_score": round(bullet_score, 4),
            "responsibility_title_score": round(title_score, 4),
            "responsibility_skill_hint_score": round(skill_hint_score, 4),
            "skill_score": round(skill_score, 4),
            "education_score": round(education_score, 4),
        }
    def achievement_score(self, exp_bullets: List[str]) -> float:
        if not exp_bullets:
            return 0.15

        verb_hits = 0
        numeric_hits = 0

        for bullet in exp_bullets:
            first = bullet.split()[0].lower() if bullet.split() else ""
            if first in ACTION_VERBS:
                verb_hits += 1
            if count_numeric_impact(bullet) > 0:
                numeric_hits += 1

        verb_part = verb_hits / max(len(exp_bullets), 1)
        num_part = numeric_hits / max(len(exp_bullets), 1)
        return min(1.0, 0.6 * verb_part + 0.4 * num_part)

    def score(self, resume_text: str, jd_text: str) -> Dict:
        section_map = self.sections.detect_sections(resume_text)
        exp_block = self.sections.extract_experience_block(resume_text)
        exp_bullets = self.extract_experience_bullets(exp_block)
        resume_education = self.sections.extract_education_lines(resume_text)
        exp_titles = self.extract_experience_titles(exp_block)

        keyword = self.keyword_report(resume_text, jd_text)
        semantic = self.semantic_score(resume_text, jd_text, exp_bullets, exp_titles, resume_education)
        achievement = self.achievement_score(exp_bullets)
        format_warnings = self.detect_format_issues(resume_text, exp_bullets)

        core_sections = ["experience", "education", "skills"]
        section_score = sum(1 for s in core_sections if section_map.get(s, False)) / len(core_sections)

        keyword_score = keyword["coverage_pct"] / 100.0
        skills_score = keyword_score
        semantic_score = semantic["overall_similarity"]
        format_score = max(0.0, 1.0 - 0.20 * len(format_warnings))

        overall = 100 * (
            0.20 * keyword_score +
            0.20 * skills_score +
            0.25 * semantic_score +
            0.10 * section_score +
            0.15 * format_score +
            0.10 * achievement
        )

        return {
            "overall_score": round(overall, 1),
            "breakdown": {
                "keyword": round(keyword_score * 100, 1),
                "skills": round(skills_score * 100, 1),
                "semantic": round(semantic_score * 100, 1),
                "sections": round(section_score * 100, 1),
                "format": round(format_score * 100, 1),
                "achievement": round(achievement * 100, 1),
            },
            "keyword_report": keyword,
            "semantic_report": semantic,
            "sections": section_map,
            "format_warnings": format_warnings,
        }






