import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
from src.facebook_publisher import FacebookReelsPublisher
from main import RedditDailyBot

@pytest.fixture
def base_config():
    return {
        "pipeline": {
            "pipeline_mode": "riddle",
            "approve_scripts": True
        },
        "facebook": {
            "enabled": True,
            "page_id": "123456789",
            "page_access_token": "EAA_test_token_XYZ",
            "caption": "Config Default Caption #test"
        }
    }

def test_publisher_disabled(base_config):
    base_config["facebook"]["enabled"] = False
    publisher = FacebookReelsPublisher(base_config)
    
    # Create dummy path
    dummy_path = Path("temp_test_video.mp4")
    
    # Should exit early without calling API
    success = publisher.publish_reel(dummy_path, "Test Caption")
    assert not success

@patch("requests.post")
def test_publisher_success(mock_post, base_config):
    # Setup mock file structure
    dummy_path = Path("temp_test_video.mp4")
    
    # Mock responses for START, UPLOAD, and FINISH phases
    mock_response_start = MagicMock()
    mock_response_start.status_code = 200
    mock_response_start.json.return_value = {
        "video_id": "987654321",
        "upload_url": "https://rupload.facebook.com/video-upload/987654321"
    }
    
    mock_response_upload = MagicMock()
    mock_response_upload.status_code = 200
    mock_response_upload.json.return_value = {"success": True}
    
    mock_response_finish = MagicMock()
    mock_response_finish.status_code = 200
    mock_response_finish.json.return_value = {"success": True, "video_id": "987654321"}
    
    # Map mocked request outcomes
    mock_post.side_effect = [mock_response_start, mock_response_upload, mock_response_finish]
    
    publisher = FacebookReelsPublisher(base_config)
    
    # Mock Path exists and stat size
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 1024
            with patch("builtins.open", mock_open(read_data=b"dummy_video_bytes")):
                success = publisher.publish_reel(dummy_path, "Dynamic Clue Caption #viral")
                
    assert success
    assert mock_post.call_count == 3
    
    # Verify START payload
    start_call = mock_post.call_args_list[0]
    assert start_call[0][0] == "https://graph.facebook.com/v19.0/123456789/video_reels"
    assert start_call[1]["data"]["upload_phase"] == "START"
    
    # Verify UPLOAD request target
    upload_call = mock_post.call_args_list[1]
    assert upload_call[0][0] == "https://rupload.facebook.com/video-upload/987654321"
    assert upload_call[1]["headers"]["offset"] == "0"
    assert upload_call[1]["headers"]["file_size"] == "1024"
    
    # Verify FINISH payload
    finish_call = mock_post.call_args_list[2]
    assert finish_call[0][0] == "https://graph.facebook.com/v19.0/123456789/video_reels"
    assert finish_call[1]["data"]["upload_phase"] == "FINISH"
    assert finish_call[1]["data"]["description"] == "Dynamic Clue Caption #viral"

@patch("requests.get")
@patch("requests.post")
def test_publisher_success_with_comment(mock_post, mock_get, base_config):
    dummy_path = Path("temp_test_video.mp4")
    
    # Mock responses for START, UPLOAD, FINISH and COMMENT posting
    mock_response_start = MagicMock()
    mock_response_start.status_code = 200
    mock_response_start.json.return_value = {
        "video_id": "987654321",
        "upload_url": "https://rupload.facebook.com/video-upload/987654321"
    }
    
    mock_response_upload = MagicMock()
    mock_response_upload.status_code = 200
    mock_response_upload.json.return_value = {"success": True}
    
    mock_response_finish = MagicMock()
    mock_response_finish.status_code = 200
    mock_response_finish.json.return_value = {"success": True, "video_id": "987654321"}
    
    mock_response_status = MagicMock()
    mock_response_status.status_code = 200
    mock_response_status.json.return_value = {
        "status": {
            "video_status": "ready",
            "processing_progress": 100
        }
    }
    mock_get.return_value = mock_response_status
    
    mock_response_comment = MagicMock()
    mock_response_comment.status_code = 200
    mock_response_comment.json.return_value = {"id": "111222333"}
    
    mock_post.side_effect = [mock_response_start, mock_response_upload, mock_response_finish, mock_response_comment]
    
    publisher = FacebookReelsPublisher(base_config)
    
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 1024
            with patch("builtins.open", mock_open(read_data=b"dummy_video_bytes")):
                success = publisher.publish_reel(dummy_path, "Dynamic Caption", "Test Pinned Comment")
                
    assert success
    assert mock_post.call_count == 4
    assert mock_get.call_count == 1
    
    # Verify comment API call parameters
    comment_call = mock_post.call_args_list[3]
    assert comment_call[0][0] == "https://graph.facebook.com/v19.0/987654321/comments"
    assert comment_call[1]["data"]["message"] == "Test Pinned Comment"
    assert comment_call[1]["data"]["access_token"] == "EAA_test_token_XYZ"

@patch("requests.post")
def test_publisher_failure_start(mock_post, base_config):
    dummy_path = Path("temp_test_video.mp4")
    
    mock_response_start = MagicMock()
    mock_response_start.status_code = 400
    mock_response_start.text = "Bad Request"
    mock_post.return_value = mock_response_start
    
    publisher = FacebookReelsPublisher(base_config)
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 1024
            success = publisher.publish_reel(dummy_path, "Test")
        
    assert not success
    assert mock_post.call_count == 1

@patch("requests.post")
def test_publisher_schedule_success(mock_post, base_config):
    dummy_path = Path("temp_test_video.mp4")
    
    # Mock responses for START, UPLOAD, and FINISH phases
    mock_response_start = MagicMock()
    mock_response_start.status_code = 200
    mock_response_start.json.return_value = {
        "video_id": "987654321",
        "upload_url": "https://rupload.facebook.com/video-upload/987654321"
    }
    
    mock_response_upload = MagicMock()
    mock_response_upload.status_code = 200
    mock_response_upload.json.return_value = {"success": True}
    
    mock_response_finish = MagicMock()
    mock_response_finish.status_code = 200
    mock_response_finish.json.return_value = {"success": True, "video_id": "987654321"}
    
    mock_post.side_effect = [mock_response_start, mock_response_upload, mock_response_finish]
    
    publisher = FacebookReelsPublisher(base_config)
    
    future_timestamp = 1783630400
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 1024
            with patch("builtins.open", mock_open(read_data=b"dummy_video_bytes")):
                success = publisher.publish_reel(
                    dummy_path, 
                    "Scheduled Caption #test", 
                    pinned_comment="Should be ignored when scheduled",
                    scheduled_publish_time=future_timestamp
                )
                
    assert success
    # Verify that we sent 3 posts (no comment post attempt because it's scheduled)
    assert mock_post.call_count == 3
    
    # Verify FINISH payload contains scheduling fields
    finish_call = mock_post.call_args_list[2]
    assert finish_call[1]["data"]["video_state"] == "SCHEDULED"
    assert finish_call[1]["data"]["scheduled_publish_time"] == str(future_timestamp)

def test_get_next_schedule_time(base_config):
    from main import RedditDailyBot
    from pathlib import Path
    
    # Setup dummy paths for test run
    history_file = Path("data/schedule_history.json")
    if history_file.exists():
        try:
            history_file.unlink()
        except Exception:
            pass
        
    # Inject missing directories for initialization
    base_config["pipeline"]["temp_dir"] = "temp"
    base_config["pipeline"]["log_dir"] = "output/logs"
    base_config["video"] = {"output_dir": "output"}
    
    bot = RedditDailyBot(base_config, MagicMock())
    
    # Calculate first slot
    ts1 = bot.get_next_schedule_time("riddle")
    assert ts1 > 0
    
    # Calculate second slot
    ts2 = bot.get_next_schedule_time("riddle")
    assert ts2 > ts1
    
    # Verify file was written
    assert history_file.exists()
    
    # Clean up test output
    if history_file.exists():
        try:
            history_file.unlink()
        except Exception:
            pass


