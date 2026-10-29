from rag_evaluation.cli import main


def test_main(capsys):
    main()

    captured = capsys.readouterr()

    assert captured.out == "RAG Evaluation Project\n"
