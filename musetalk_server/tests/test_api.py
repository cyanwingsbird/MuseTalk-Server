import pytest
from fastapi.testclient import TestClient
from musetalk_server.app import app

# Tests run without GPU — verify API contract, validation, and error handling.
# Full integration tests (preprocessing, inference) require models and GPU.

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health & System
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_schema(self):
        data = client.get("/health").json()
        assert data["status"] == "running"
        assert isinstance(data["models"]["loaded"], bool)
        assert isinstance(data["models"]["device"], str)
        assert isinstance(data["loaded_avatars"], list)


# ---------------------------------------------------------------------------
# Avatars - Listing
# ---------------------------------------------------------------------------

class TestListAvatars:
    def test_returns_200_with_list(self):
        response = client.get("/avatars")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


# ---------------------------------------------------------------------------
# Avatars - Preprocessing (avatar_id validation)
# ---------------------------------------------------------------------------

class TestPreprocessValidation:
    """Verify avatar_id is validated before any filesystem or model work."""

    @pytest.mark.parametrize("bad_id", [
        "../etc/passwd",
        "../../secret",
        "foo/bar",
        "foo\\bar",
        "a" * 65,          # exceeds 64-char limit
        "hello world",      # spaces
        "avatar<script>",   # XSS attempt
    ])
    def test_rejects_invalid_avatar_id(self, bad_id):
        response = client.post(
            "/avatars/preprocess",
            data={"avatar_id": bad_id, "bbox_shift": "0"},
            files={"video_file": ("test.mp4", b"fake video", "video/mp4")},
        )
        assert response.status_code == 400
        assert "Invalid avatar_id" in response.json()["detail"]

    def test_rejects_empty_avatar_id(self):
        """Empty string is rejected by FastAPI's Form(...) required validation."""
        response = client.post(
            "/avatars/preprocess",
            data={"avatar_id": "", "bbox_shift": "0"},
            files={"video_file": ("test.mp4", b"fake video", "video/mp4")},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("good_id", [
        "my_avatar",
        "avatar-01",
        "TestAvatar123",
        "a",
        "a" * 64,          # exactly at limit
    ])
    def test_accepts_valid_avatar_id_format(self, good_id):
        """Valid format should pass validation (will fail later due to no models, not 400)."""
        response = client.post(
            "/avatars/preprocess",
            data={"avatar_id": good_id, "bbox_shift": "0"},
            files={"video_file": ("test.mp4", b"fake video", "video/mp4")},
        )
        # Should NOT be 400 (validation passed); will be 500 because models aren't loaded
        assert response.status_code != 400


# ---------------------------------------------------------------------------
# Inference - Stream (missing avatar & validation)
# ---------------------------------------------------------------------------

class TestStreamInference:
    def test_missing_avatar_returns_404(self):
        response = client.post(
            "/inference/stream/nonexistent_avatar",
            files={"audio_file": ("test.wav", b"fake audio", "audio/wav")},
        )
        assert response.status_code == 404

    def test_rejects_too_long_avatar_id(self):
        """IDs exceeding 64 chars are caught by our validation (400)."""
        response = client.post(
            f"/inference/stream/{'a' * 65}",
            files={"audio_file": ("test.wav", b"fake audio", "audio/wav")},
        )
        assert response.status_code == 400
        assert "Invalid avatar_id" in response.json()["detail"]

    @pytest.mark.parametrize("bad_id", [
        "../etc/passwd",
        "foo/bar",
        "",
    ])
    def test_path_traversal_blocked_by_routing(self, bad_id):
        """Slashes and empty strings in URL path never reach our handler (blocked by routing)."""
        response = client.post(
            f"/inference/stream/{bad_id}",
            files={"audio_file": ("test.wav", b"fake audio", "audio/wav")},
        )
        assert response.status_code in (404, 405, 307)


# ---------------------------------------------------------------------------
# Inference - Batch (missing avatar & validation)
# ---------------------------------------------------------------------------

class TestBatchInference:
    def test_missing_avatar_returns_404(self):
        response = client.post(
            "/inference/batch/nonexistent_avatar",
            files={"audio_file": ("test.wav", b"fake audio", "audio/wav")},
        )
        assert response.status_code == 404

    def test_rejects_too_long_avatar_id(self):
        """IDs exceeding 64 chars are caught by our validation (400)."""
        response = client.post(
            f"/inference/batch/{'a' * 65}",
            files={"audio_file": ("test.wav", b"fake audio", "audio/wav")},
        )
        assert response.status_code == 400
        assert "Invalid avatar_id" in response.json()["detail"]

    @pytest.mark.parametrize("bad_id", [
        "../etc/passwd",
        "foo/bar",
        "",
    ])
    def test_path_traversal_blocked_by_routing(self, bad_id):
        """Slashes and empty strings in URL path never reach our handler (blocked by routing)."""
        response = client.post(
            f"/inference/batch/{bad_id}",
            files={"audio_file": ("test.wav", b"fake audio", "audio/wav")},
        )
        assert response.status_code in (404, 405, 307)
