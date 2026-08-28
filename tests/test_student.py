import pytest

torch = pytest.importorskip("torch")


def test_student_embedding_shape_and_norm() -> None:
    from wam_drones.student.model import StudentEmbeddingModel

    model = StudentEmbeddingModel().eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 160, 160))
    assert output.shape == (2, 64)
    assert torch.allclose(output.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_scene_split_is_deterministic_and_disjoint(tmp_path: object) -> None:
    from pathlib import Path

    from wam_drones.student.data import DistillationSample, deterministic_split

    np = pytest.importorskip("numpy")

    samples = [
        DistillationSample(
            Path(str(tmp_path)) / f"{index}.jpg",
            f"scene-{index}",
            "bicycle" if index < 5 else None,
            np.ones(512, dtype=np.float32),
        )
        for index in range(10)
    ]
    first_train, first_eval = deterministic_split(samples, 42, 0.2)
    second_train, second_eval = deterministic_split(samples, 42, 0.2)

    assert [sample.source_id for sample in first_train] == [
        sample.source_id for sample in second_train
    ]
    assert [sample.source_id for sample in first_eval] == [
        sample.source_id for sample in second_eval
    ]
    assert {sample.source_id for sample in first_train}.isdisjoint(
        sample.source_id for sample in first_eval
    )


def test_static_onnx_parity(tmp_path: object) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from pathlib import Path

    from wam_drones.student.model import StudentEmbeddingModel

    destination = Path(str(tmp_path)) / "student.onnx"
    model = StudentEmbeddingModel().eval()
    example = torch.randn(1, 3, 160, 160)
    torch.onnx.export(model, example, destination, opset_version=17, dynamo=False)
    expected = model(example).detach().numpy()
    session = onnxruntime.InferenceSession(str(destination))
    actual = session.run(None, {session.get_inputs()[0].name: example.numpy()})[0]
    assert actual.shape == (1, 64)
    assert abs((actual**2).sum() - 1.0) < 1e-4
    assert abs(actual - expected).max() < 1e-5
