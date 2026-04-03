from __future__ import annotations

from app.rag.schemas import AudienceType


STUDENT_TOPIC_HINTS = {
    'citation': ['citation', 'reference', 'referencing', 'apa', 'harvard', 'mla'],
    'structure': ['structure', 'essay', 'introduction', 'conclusion', 'paragraph'],
    'clarity': ['clarity', 'coherence', 'flow', 'academic writing'],
    'integrity': ['plagiarism', 'integrity', 'original work'],
}

PROFESSOR_TOPIC_HINTS = {
    'rubric': ['rubric', 'criterion', 'descriptor', 'band'],
    'policy': ['policy', 'assessment', 'regulation', 'grade'],
    'moderation': ['moderation', 'second marking', 'consistency'],
    'feedback': ['feedback', 'wording', 'comment bank', 'feedforward'],
}


def detect_topic(query: str, audience: AudienceType) -> str | None:
    query_l = query.lower()
    source = STUDENT_TOPIC_HINTS if audience == 'student' else PROFESSOR_TOPIC_HINTS
    for topic, hints in source.items():
        if any(h in query_l for h in hints):
            return topic
    return None
