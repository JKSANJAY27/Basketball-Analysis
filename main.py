from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import (PlayerTracksDrawer, BallTracksDrawer, TeamBallControlDrawer, PassInterceptionDrawer, CourtKeypointDrawer, TacticalViewDrawer)
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector
from pass_and_interception_detector import PassAndInterceptionDetector
from court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter import TacticalViewConverter

def main():
    video_frames = read_video("input_videos/video_2.mp4")

    player_tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/ball_detector_model.pt")
    court_keypoint_detector = CourtKeypointDetector("models/court_keypoint_detector.pt")

    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/player_track_stubs.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/ball_track_stubs.pkl")
    court_keypoints = court_keypoint_detector.get_court_keypoints(video_frames, read_from_stub=True, stub_path="stubs/court_key_points_stubs.pkl")
    
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames, player_tracks, read_from_stub=True, stub_path="stubs/player_assignment_stub.pkl")

    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_acquisition, player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_acquisition, player_assignment)

    tactical_view_converter = TacticalViewConverter(court_image_path = "./images/basketball_court.png")
    tactical_view_converter.validate_keypoints(court_keypoints)
    tactical_player_positions = tactical_view_converter.transform_players_to_tactical_view(court_keypoints, player_tracks)

    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_interception_drawer = PassInterceptionDrawer()
    court_keypoint_drawer = CourtKeypointDrawer()
    tactical_view_drawer = TacticalViewDrawer()

    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment, ball_acquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_acquisition)
    output_video_frames = pass_interception_drawer.draw(output_video_frames, passes, interceptions)
    output_video_frames = court_keypoint_drawer.draw(output_video_frames, court_keypoints)
    output_video_frames = tactical_view_drawer.draw(output_video_frames, tactical_view_converter.court_image_path, tactical_view_converter.width, tactical_view_converter.height, tactical_view_converter.key_points, tactical_player_positions, player_assignment, ball_acquisition)

    save_video(output_video_frames, "output_videos/output_video.avi")
    
if __name__ == "__main__":
    main()