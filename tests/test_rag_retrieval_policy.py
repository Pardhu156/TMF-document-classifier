from src.rag.retrieval_policy import MASTER_SOURCE_TYPE, build_retrieval_plan, fuzzy_file_matches


DOCUMENTS = [
    {"document_id": "master_a91b72", "file_name": "Cancer_Protocol_v2.pdf", "source_type": "MASTER_DATA"},
    {"document_id": "upload_2", "file_name": "Breast_Cancer_CSR.docx", "source_type": "PREDICT_UPLOAD"},
]


def test_explicit_document_id_has_highest_priority() -> None:
    plan = build_retrieval_plan(
        question="Explain the objective from Cancer Protocol file",
        documents=DOCUMENTS,
        document_id="master_a91b72",
        scope="all",
    )

    assert plan.filters == {"document_id": "master_a91b72"}
    assert plan.retrieval_scope == "document"


def test_filename_match_restricts_to_matched_file() -> None:
    plan = build_retrieval_plan(
        question="Explain objective from Cancer Protocol file",
        documents=DOCUMENTS,
    )

    assert plan.filters == {"file_name": "Cancer_Protocol_v2.pdf"}
    assert plan.matched_file_name == "Cancer_Protocol_v2.pdf"
    assert plan.retrieval_scope == "file"


def test_default_scope_retrieves_only_master_data() -> None:
    plan = build_retrieval_plan(
        question="What is the study objective?",
        documents=DOCUMENTS,
    )

    assert plan.filters == {"source_type": MASTER_SOURCE_TYPE}
    assert plan.retrieval_scope == "master"


def test_scope_all_has_no_source_filter() -> None:
    plan = build_retrieval_plan(
        question="What is the study objective?",
        documents=DOCUMENTS,
        scope="all",
    )

    assert plan.filters == {}
    assert plan.retrieval_scope == "all"


def test_scope_verified_filters_verification_status() -> None:
    plan = build_retrieval_plan(
        question="What is the study objective?",
        documents=DOCUMENTS,
        scope="verified",
    )

    assert plan.filters == {"verification_status": "verified"}


def test_ambiguous_filename_match_requires_clarification() -> None:
    matched_file, candidates = fuzzy_file_matches(
        "Summarize cancer protocol",
        [
            {"file_name": "Cancer_Protocol_v1.pdf"},
            {"file_name": "Cancer_Protocol_v2.pdf"},
        ],
    )

    assert matched_file is None
    assert len(candidates) == 2
