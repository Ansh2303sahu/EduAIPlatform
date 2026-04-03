from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "knowledge"

DOCS = {
    "backend/knowledge/student/writing/essay_structure.md": """A strong academic essay normally includes an introduction, a logically organised main body, and a conclusion. The introduction should identify the topic, define the focus of the discussion, and present the central argument or purpose. The main body should be organised into paragraphs, with each paragraph focusing on one clear idea supported by explanation and evidence. The conclusion should summarise the argument without introducing new evidence.

Essay structure should help the reader follow the line of reasoning. Clear organisation improves coherence and makes the work easier to evaluate. Students should ensure that each paragraph contributes directly to the overall argument rather than adding loosely connected description.
""",
    "backend/knowledge/student/writing/paragraph_development.md": """A well-developed paragraph usually begins with a topic sentence that signals the main point. This should be followed by explanation, evidence, and analysis. Evidence may include examples, research findings, or academic sources. Analysis should explain why the evidence matters and how it supports the paragraph's point.

Paragraphs should not simply list information. They should show progression of thought. Good paragraph development improves clarity, coherence, and academic quality.
""",
    "backend/knowledge/student/referencing/apa7_basics.md": """APA 7 referencing requires consistency in author, year, title, and source formatting. Reference list entries should be arranged alphabetically by author surname. Journal references should usually include the author, year, article title, journal title, volume, issue where available, page range, and DOI or URL if required.

Students should ensure that every in-text citation has a matching reference list entry. Missing details and inconsistent formatting reduce academic credibility.
""",
    "backend/knowledge/student/referencing/in_text_citation_rules.md": """In-text citations in APA 7 generally include the author's surname and year of publication. Direct quotations should also include a page number where available. Paraphrased material still requires citation because the idea is taken from another source.

Students should use citations whenever they summarise, paraphrase, or directly quote another author's work. Referencing is necessary to acknowledge sources and avoid plagiarism.
""",
    "backend/knowledge/student/critical_thinking/critical_analysis.md": """Critical analysis involves more than describing information. It requires the student to evaluate evidence, identify strengths and weaknesses, compare viewpoints, and explain the significance of findings. Strong critical writing shows why a source or argument matters rather than simply repeating what it says.

Students should connect analysis directly to the question. Effective critical analysis often includes comparison, judgement, interpretation, and clear reasoning.
""",
    "backend/knowledge/student/critical_thinking/compare_and_evaluate.md": """Comparing viewpoints is an important part of academic evaluation. Students should identify similarities and differences between theories, methods, or findings. Evaluation involves judging the quality or usefulness of the evidence presented.

High-quality comparison does not stop at saying two authors disagree. It explains why they differ and which interpretation is more convincing in relation to the question.
""",
    "backend/knowledge/student/academic_integrity/plagiarism_guidance.md": """Plagiarism occurs when a student presents another person's words, ideas, or work as their own without appropriate acknowledgement. This includes copying direct text without quotation and citation, or closely paraphrasing without attribution.

Academic integrity requires honesty, proper referencing, and independent work. Students should keep track of their sources and cite them consistently throughout the assignment.
""",
    "backend/knowledge/student/academic_integrity/paraphrasing_guidance.md": """Effective paraphrasing means rewriting source material fully in your own words and sentence structure while keeping the original meaning. Changing only a few words is not enough. A paraphrase must still be followed by a citation because the underlying idea comes from another source.

Good paraphrasing demonstrates understanding. It should sound natural within the student's own writing style and fit the surrounding argument.
""",
    "backend/knowledge/student/research_support/source_quality.md": """Source quality should be evaluated before evidence is used in academic work. Reliable sources are often peer-reviewed journal articles, academic books, official reports, and credible institutional publications. Students should consider the author's expertise, publication venue, date, and relevance.

Using weak or unreliable sources can reduce the quality of an assignment. Strong source selection supports stronger argumentation.
""",
    "backend/knowledge/student/research_support/using_evidence.md": """Evidence should be integrated into academic writing with explanation and analysis. Students should avoid dropping quotations or findings into a paragraph without showing how they support the point being made. Evidence is most effective when it is introduced clearly, cited correctly, and then interpreted.

Academic writing should use evidence selectively and purposefully. The aim is not to include as many sources as possible, but to use relevant sources effectively.
""",
    "backend/knowledge/professor/rubrics/grading_descriptors.md": """Grading descriptors should reflect published academic standards. Higher-level work is usually characterised by clarity, relevance, strong structure, accurate use of evidence, and analytical depth. Lower-level work may show weak structure, limited understanding, descriptive writing, or inadequate support.

Feedback should explain how the work aligns with these descriptors. Clear alignment improves fairness and transparency.
""",
    "backend/knowledge/professor/rubrics/rubric_alignment.md": """Assessment feedback should be aligned with rubric criteria. Comments should refer to the relevant dimensions of performance rather than offering unrelated impressions. Where possible, feedback should explain why the work meets, partially meets, or fails to meet a criterion.

Rubric alignment improves consistency across markers and helps students understand the basis of the judgement.
""",
    "backend/knowledge/professor/rubrics/high_distinction_descriptors.md": """High distinction level work is typically characterised by strong relevance to the question, clear and logical structure, precise academic writing, and consistent use of appropriate evidence. It demonstrates analytical depth rather than simple description. Arguments are usually well supported and clearly developed across the full response.

Markers should distinguish high distinction work by its coherence, critical engagement, and ability to connect evidence directly to the assessment criteria.
""",
    "backend/knowledge/professor/rubrics/analytical_depth_indicators.md": """Analytical depth is shown when the student moves beyond description and explains why evidence matters. Strong analytical work compares viewpoints, evaluates strengths and weaknesses, and links ideas directly to the question. Limited analytical depth is shown when the work mainly summarises sources without interpretation.

Markers should use analytical depth as an indicator of higher-level performance when applying rubric criteria.
""",
    "backend/knowledge/professor/marking_policy/marking_consistency.md": """Marking consistency is essential for fairness. Similar quality work should receive similar judgement across different scripts and markers. Markers should apply shared criteria consistently and avoid letting isolated strengths or weaknesses distort the overall decision.

Consistency can be improved through calibration, moderation, and reference to agreed descriptors.
""",
    "backend/knowledge/professor/marking_policy/borderline_work.md": """Borderline work should be judged carefully against the published criteria. Markers should identify whether the work more strongly meets the level above or below, using evidence from structure, argument quality, source use, and relevance. Borderline decisions should be justified clearly in notes.

A cautious and evidence-based approach is important where the performance sits between two adjacent bands.
""",
    "backend/knowledge/professor/marking_policy/criterion_level_justification.md": """Criterion-level justification means explaining why performance meets, partially meets, or fails to meet each assessment criterion. This helps ensure that marks are defensible and transparent. Justification should refer to features of the script such as structure, argument quality, evidence, relevance, and clarity.

Strong criterion-level justification supports consistency across markers and moderation.
""",
    "backend/knowledge/professor/moderation/moderation_notes.md": """Moderation notes should explain the reasoning behind significant marking decisions, especially where a script is borderline, inconsistent, or unusually strong or weak. Notes should be concise but specific enough to support transparency and later review.

Good moderation notes focus on evidence from the script and connection to criteria rather than vague impressions.
""",
    "backend/knowledge/professor/moderation/second_marking.md": """Second marking helps strengthen fairness and reliability. Where a second marker is involved, differences in judgement should be resolved through reference to the rubric and evidence in the work. The aim is not simply compromise but a justified academic decision.

Useful second-marking practice includes identifying the exact criteria where disagreement exists and explaining the reason for the final mark.
""",
    "backend/knowledge/professor/moderation/disagreement_resolution.md": """When markers disagree, moderation should focus on the rubric and direct evidence from the script. The purpose is not to average opinions mechanically, but to reach a justified academic decision. Differences should be discussed criterion by criterion, especially where the script is borderline between adjacent bands.

Clear disagreement resolution improves fairness, transparency, and consistency.
""",
    "backend/knowledge/professor/feedback_templates/actionable_feedback.md": """Effective feedback should be specific, balanced, and actionable. It should identify strengths, explain weaknesses, and suggest realistic next steps for improvement. Vague comments such as "be clearer" are less helpful than comments that explain what should be clarified and where.

Actionable feedback helps students understand how to improve future work rather than only explaining what was wrong.
""",
    "backend/knowledge/professor/feedback_templates/balanced_feedback.md": """Balanced feedback recognises both strengths and areas for development. Overly negative feedback may be discouraging, while overly general praise may not help learning. Good feedback should show what the student did well and what should be improved, with both linked to the assessment criteria.

Balanced comments support constructive academic development.
""",
    "backend/knowledge/professor/feedback_templates/weak_vs_strong_feedback.md": """Weak feedback is vague, general, or disconnected from the assessment criteria. Comments such as "be clearer" or "improve analysis" provide little guidance if they do not explain where and how improvement is needed. Strong feedback is specific, linked to evidence from the script, and gives actionable next steps.

Effective feedback should make the basis of judgement understandable and support improvement in future work.
""",
    "backend/knowledge/professor/academic_quality/evidence_quality_expectations.md": """High-quality academic work usually uses relevant and credible evidence effectively. Evidence should be integrated into the student's argument rather than added without explanation. Weak evidence use may include unsupported claims, irrelevant sources, or limited interpretation of research.

Markers should identify whether evidence is merely present or whether it is used analytically to strengthen the response.
""",
    "backend/knowledge/professor/academic_quality/high_quality_writing.md": """High-quality academic writing is usually clear, well organised, relevant to the question, and supported by appropriate evidence. It demonstrates analytical thinking, logical progression, and accurate academic style. Strong work also integrates sources effectively rather than relying on unsupported claims.

Markers should distinguish between descriptive competence and genuine analytical quality.
""",
    "backend/knowledge/professor/academic_quality/common_weaknesses.md": """Common weaknesses in academic writing include weak structure, poor paragraph control, limited analysis, insufficient evidence, and unclear relevance to the question. Referencing errors and descriptive writing are also frequent issues.

Feedback on common weaknesses should be specific enough to guide improvement and should identify the academic skill that needs further development.
""",
}


def resolve_doc_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.parts[:2] == ("backend", "knowledge"):
        path = Path(*path.parts[2:])
    return BASE_DIR / path


for path_str, content in DOCS.items():
    path = resolve_doc_path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")

print(f"Created {len(DOCS)} seed knowledge files.")
