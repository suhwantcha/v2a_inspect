import argparse
import json
import logging
import os
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def process_file(json_path: Path, video_dir: Path, output_dir: Path, env: Environment) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    scene_analysis = data.get("scene_analysis", data)
    video_id = json_path.stem
    
    # Locate original video
    video_path = None
    for ext in ['.mp4', '.avi', '.mov', '.webm']:
        potential = video_dir / f"{video_id}{ext}"
        if potential.exists():
            video_path = potential
            break
            
    # Simply copy the original video file to use HTML5 fragment URLs (#t=start,end)
    videos_out_dir = output_dir / "videos"
    os.makedirs(videos_out_dir, exist_ok=True)
    
    out_video_path = ""
    if video_path:
        out_video_path = videos_out_dir / video_path.name
        if not out_video_path.exists():
            logger.info(f"Copying video {video_path.name}...")
            shutil.copy2(video_path, out_video_path)
    else:
        logger.warning(f"Video file for {video_id} not found in {video_dir}. Video playbacks will not work.")

    # Render per_video template
    template = env.get_template("per_video.html.jinja2")
    html_output = template.render(
        video_id=video_id,
        data=data,
        scene_analysis=scene_analysis,
        video_filename=video_path.name if video_path else ""
    )
    
    out_html_path = output_dir / "per_video" / f"{video_id}.html"
    os.makedirs(out_html_path.parent, exist_ok=True)
    with open(out_html_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
        
    logger.info(f"Generated {out_html_path}")
    
    # Return stats for index
    return {
        "video_id": video_id,
        "total_duration": scene_analysis.get("total_duration", 0),
        "num_scenes": len(scene_analysis.get("scenes", [])),
        "num_tracks": len(data.get("raw_tracks", [])),
        "num_groups": len(data.get("groups", []))
    }

def generate_index(stats: list[dict], output_dir: Path, env: Environment) -> None:
    template = env.get_template("index.html.jinja2")
    
    # Averages
    avg_scenes = sum(s["num_scenes"] for s in stats) / max(1, len(stats))
    avg_tracks = sum(s["num_tracks"] for s in stats) / max(1, len(stats))
    avg_groups = sum(s["num_groups"] for s in stats) / max(1, len(stats))
    
    html_output = template.render(
        stats=stats,
        avg_scenes=f"{avg_scenes:.1f}",
        avg_tracks=f"{avg_tracks:.1f}",
        avg_groups=f"{avg_groups:.1f}"
    )
    
    out_path = output_dir / "index.html"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    logger.info(f"Generated index at {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate HTML reports and video clips from JSON.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Dir with JSON results")
    parser.add_argument("--videos-dir", type=Path, default=Path("."), help="Dir with original video files")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output dir for index and per_video")
    args = parser.parse_args()
    
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(templates_dir))
    
    stats = []
    
    if not args.results_dir.exists():
        logger.error(f"Results dir {args.results_dir} does not exist.")
        if Path("grouped_result.json").exists():
            logger.info("Found grouped_result.json, processing it.")
            stats.append(process_file(Path("grouped_result.json"), args.videos_dir, args.output_dir, env))
    else:
        for json_file in args.results_dir.glob("*.json"):
            stats.append(process_file(json_file, args.videos_dir, args.output_dir, env))
            
    if stats:
        generate_index(stats, args.output_dir, env)
    else:
        logger.warning("No JSON results processed.")

if __name__ == "__main__":
    main()
