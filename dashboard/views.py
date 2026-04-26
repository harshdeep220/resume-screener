import os
import json
import time
from pathlib import Path

from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# Add root src path
import sys
sys.path.append(str(settings.BASE_DIR))

from src.extractor import extract_text
from src.jd_parser import parse_jd
from src.resume_parser import parse_resume
from src.nlp_engine import score_resumes
from src.ai_scorer import score_resume, _save_cache
from src.scoring_engine import compute_final_scores
from google import genai

def index(request):
    return render(request, "dashboard/index.html")

@csrf_exempt
def run_pipeline(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)

    try:
        config_path = settings.BASE_DIR / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        nlp_weight = float(request.POST.get('nlp_weight', config.get("nlp_weight", 0.4)))
        ai_weight = float(request.POST.get('ai_weight', config.get("ai_weight", 0.6)))
        model_name = config.get("model", "gemini-3.1-flash-lite-preview")
        
        jd_file = request.FILES.get("jd_file")
        jd_text_input = request.POST.get("jd_text")
        
        jd_dir = settings.MEDIA_ROOT / "jds"
        resumes_dir = settings.MEDIA_ROOT / "resumes"
        
        # Clear out old resumes
        for filename in os.listdir(resumes_dir):
            file_path = os.path.join(resumes_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception:
                pass
        
        jd_text = ""
        jd_profile = None
        
        if jd_file:
            jd_path = jd_dir / jd_file.name
            with open(jd_path, "wb+") as f:
                for chunk in jd_file.chunks():
                    f.write(chunk)
            _, jd_text = extract_text(jd_path)
        elif jd_text_input:
            jd_text = jd_text_input
            
        if not jd_text:
            return JsonResponse({"error": "Job Description text or file is required."}, status=400)
            
        jd_profile = parse_jd(jd_text)
        
        resume_files = request.FILES.getlist("resume_files")
        if not resume_files:
            return JsonResponse({"error": "At least one resume file must be uploaded."}, status=400)
            
        resumes = []
        for rf in resume_files:
            r_path = resumes_dir / rf.name
            with open(r_path, "wb+") as f:
                for chunk in rf.chunks():
                    f.write(chunk)
            
            _, r_text = extract_text(r_path)
            if r_text:
                resumes.append((rf.name, r_text))
                
        if not resumes:
            return JsonResponse({"error": "Could not extract text from the provided resumes."}, status=400)
            
        resume_profiles = [parse_resume(t, f) for f, t in resumes]
        
        nlp_results = score_resumes(
            jd_skills=jd_profile.required_skills,
            jd_text=jd_profile.raw_text,
            resume_skills_list=[rp.skills for rp in resume_profiles],
            resume_texts=[rp.raw_text for rp in resume_profiles],
        )
        
        api_key = os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key) if api_key and api_key != "your_key_here" else None
        cache_dict = {}
        ai_results = []
        api_delay = config.get("api_delay_seconds", 2)
        
        for i, rp in enumerate(resume_profiles):
            result = score_resume(
                jd_text=jd_profile.raw_text,
                resume_text=rp.raw_text,
                model_name=model_name,
                api_delay=api_delay,
                _cache=cache_dict,
                _client=client,
            )
            ai_results.append(result)
            if i < len(resume_profiles) - 1:
                time.sleep(api_delay)
                
        _save_cache(cache_dict)
        
        ranked = compute_final_scores(
            filenames=[rp.filename for rp in resume_profiles],
            candidate_names=[rp.candidate_name for rp in resume_profiles],
            nlp_scores=[nr.nlp_score for nr in nlp_results],
            ai_scores=[float(ar["score"]) for ar in ai_results],
            ai_rationales=[ar["rationale"] for ar in ai_results],
            skill_matches_list=[nr.skill_matches for nr in nlp_results],
            skill_gaps_list=[nr.skill_gaps for nr in nlp_results],
            nlp_weight=nlp_weight,
            ai_weight=ai_weight,
        )
        
        response_data = []
        for index, r in enumerate(ranked, start=1):
            response_data.append({
                "rank": index,
                "name": r.candidate_name,
                "file": r.filename,
                "nlp": r.nlp_score,
                "ai": r.ai_score,
                "final": r.final_score,
                "rationale": r.rationale,
                "matched": list(r.skill_matches),
                "missing": list(r.skill_gaps)
            })
            
        return JsonResponse({"results": response_data})
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
