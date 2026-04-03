"""
DDR Pipeline  —  Fixed & Improved
Run:
    python ddr_fixed.py --inspection "Sample Report.pdf" \
                        --thermal    "Thermal Images.pdf" \
                        --output     "Final_DDR.pdf"

Set your key:  export GROQ_API_KEY=your_key
               (or put it in a .env file)
"""

import argparse, json, os, re, sys, warnings
import pdfplumber
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether
)

warnings.filterwarnings("ignore")

# ── STEP 1: Extract text ─────────────────────────────────────────────────────

def extract_text(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)

def extract_thermal_readings(path: str) -> list[dict]:
    """
    Thermal PDF uses UTF-16 encoding — pdfplumber gets nothing.
    pypdf + strip null bytes works correctly.
    Returns one dict per scan page.
    """
    readings = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        raw = (page.extract_text() or "").replace("\x00", "")
        hot  = re.search(r"Hotspot\s*:\s*([\d.]+)", raw)
        cold = re.search(r"Coldspot\s*:\s*([\d.]+)", raw)
        emis = re.search(r"Emissivity\s*:\s*([\d.]+)", raw)
        date = re.search(r"(\d{2}/\d{2}/\d{2,4})", raw)
        if hot and cold:
            h, c = float(hot.group(1)), float(cold.group(1))
            readings.append({
                "scan":       i + 1,
                "hotspot":    h,
                "coldspot":   c,
                "delta_t":    round(h - c, 2),
                "emissivity": float(emis.group(1)) if emis else 0.94,
                "date":       date.group(1) if date else "27/09/22",
            })
    return readings


# ── STEP 2: Extract images ───────────────────────────────────────────────────

def get_good_images(pdf_path: str, prefix: str, min_px: int = 300) -> list[str]:
    """
    Extract images via pdfimages, return paths of real photos (skip tiny icons).
    Falls back gracefully if pdfimages not installed.
    """
    import subprocess, tempfile
    out_dir = tempfile.mkdtemp()
    out_prefix = os.path.join(out_dir, prefix)
    r = subprocess.run(
        ["pdfimages", "-png", pdf_path, out_prefix],
        capture_output=True
    )
    if r.returncode != 0:
        print("  ⚠ pdfimages not found — images will show 'Not Available'")
        return []

    paths = []
    for f in sorted(Path(out_dir).glob(f"{prefix}*.png")):
        try:
            w, h = Image.open(f).size
            if w >= min_px and h >= min_px:
                paths.append(str(f))
        except Exception:
            pass
    return paths


# ── STEP 3: Call Groq AI ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a building diagnostics AI. Analyze inspection and thermal data.
Output ONLY valid JSON — no markdown, no explanation.

CRITICAL RULES:
- Never invent facts. Only use information in the documents.
- Write "Not Available" for anything missing.
- thermal_findings must include actual Hotspot/Coldspot numbers if present in thermal data.
- missing_information must list everything that could not be found."""

def call_ai(insp_text: str, thermal_summary: str, api_key: str) -> dict:
    client = Groq(api_key=api_key)

    prompt = f"""
INSPECTION REPORT:
{insp_text}

THERMAL READINGS (extracted):
{thermal_summary}

Return this exact JSON structure:
{{
  "property_address": "string or Not Available",
  "customer_name": "string or Not Available",
  "inspection_date": "string or Not Available",
  "property_summary": "2-3 sentence plain-English overview of all issues",
  "observations": [
    {{
      "area": "string",
      "damage_observed": "string — visible symptoms",
      "source_identified": "string — what is causing it",
      "thermal_findings": "string — include Hotspot/Coldspot/Delta-T numbers if available, else Not Available",
      "severity": "Critical|High|Medium|Low",
      "photo_indices": ["Array of integers (e.g., [1, 2, 3]) representing the Photo numbers explicitly mentioned in the text for this area"]
    }}
  ],
  "root_causes": [
    {{"issue": "string", "cause": "string"}}
  ],
  "severity_summary": "string — overall severity with reasoning",
  "actions": [
    {{"priority": "Immediate|Short-term|Long-term", "area": "string", "task": "string"}}
  ],
  "additional_notes": "string",
  "missing_information": ["list of unavailable items"]
}}
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    raw = re.sub(r"^```json\s*", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    return json.loads(raw)


# ── STEP 4: Render PDF ───────────────────────────────────────────────────────

C_DARK   = colors.HexColor("#1A1A2E")
C_ACCENT = colors.HexColor("#E8A020")
C_LIGHT  = colors.HexColor("#F5F5F5")
C_RED    = colors.HexColor("#C0392B")
C_ORANGE = colors.HexColor("#E67E22")
C_YELLOW = colors.HexColor("#F39C12")
C_GREEN  = colors.HexColor("#27AE60")
C_BLUE   = colors.HexColor("#EBF5FB")
C_BLBDR  = colors.HexColor("#2980B9")
C_GRAY   = colors.HexColor("#666666")
WHITE    = colors.white

SEV_COLOR = {
    "Critical": C_RED, "High": C_ORANGE,
    "Medium": C_YELLOW, "Low": C_GREEN,
}

def sev_badge(level: str) -> Table:
    bg = SEV_COLOR.get(level, C_GRAY)
    s  = ParagraphStyle("b", fontName="Helvetica-Bold", fontSize=8,
                        textColor=WHITE, alignment=TA_CENTER, leading=10)
    return Table([[Paragraph(level.upper(), s)]], colWidths=[22*mm],
                 style=[("BACKGROUND",(0,0),(-1,-1),bg),
                        ("TOPPADDING",(0,0),(-1,-1),3),
                        ("BOTTOMPADDING",(0,0),(-1,-1),3)])

def thermal_box(text: str) -> Table:
    s = ParagraphStyle("t", fontName="Helvetica-Oblique", fontSize=9,
                       textColor=colors.HexColor("#1A5276"), leading=13)
    return Table([[Paragraph(f"🌡 {text}", s)]], colWidths=[165*mm],
                 style=[("BACKGROUND",(0,0),(-1,-1),C_BLUE),
                        ("LINEBEFORE",(0,0),(0,-1),3,C_BLBDR),
                        ("LEFTPADDING",(0,0),(-1,-1),8),
                        ("TOPPADDING",(0,0),(-1,-1),5),
                        ("BOTTOMPADDING",(0,0),(-1,-1),5)])

def section_header(num: str, title: str) -> Table:
    s = ParagraphStyle("sh", fontName="Helvetica-Bold", fontSize=10,
                       textColor=WHITE, leading=13)
    return Table([[Paragraph(f"{num}  {title.upper()}", s)]], colWidths=[175*mm],
                 style=[("BACKGROUND",(0,0),(-1,-1),C_DARK),
                        ("LEFTPADDING",(0,0),(-1,-1),10),
                        ("TOPPADDING",(0,0),(-1,-1),6),
                        ("BOTTOMPADDING",(0,0),(-1,-1),6)])

def add_image(path: str, max_w=80*mm, max_h=60*mm):
    """Return RLImage scaled to fit, or None."""
    try:
        img = Image.open(path)
        w, h = img.size
        scale = min(max_w / w, max_h / h)
        return RLImage(path, width=w*scale, height=h*scale)
    except Exception:
        return None

def render(ddr: dict, insp_imgs: list[str], therm_imgs: list[str], output: str):
    import datetime
    doc = SimpleDocTemplate(output, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    body  = ParagraphStyle("body",  fontName="Helvetica",      fontSize=9.5, leading=15, textColor=C_DARK, spaceAfter=4)
    label = ParagraphStyle("lbl",   fontName="Helvetica-Bold", fontSize=9,   leading=13, textColor=C_GRAY)
    atitle= ParagraphStyle("at",    fontName="Helvetica-Bold", fontSize=10,  leading=14, textColor=C_DARK, spaceAfter=2)
    small = ParagraphStyle("sm",    fontName="Helvetica",      fontSize=8.5, leading=13, textColor=C_GRAY)
    foot  = ParagraphStyle("ft",    fontName="Helvetica",      fontSize=7.5, leading=11, textColor=C_GRAY, alignment=TA_CENTER)

    sp = lambda h=6: Spacer(1, h)
    hr = lambda: HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#DDDDDD"))

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    cover_s = ParagraphStyle("cov", fontName="Helvetica-Bold", fontSize=24,
                             textColor=WHITE, leading=30)
    sub_s   = ParagraphStyle("sub", fontName="Helvetica", fontSize=11,
                             textColor=C_ACCENT, leading=16)
    story.append(Table([
        [Paragraph("Detailed Diagnostic Report", cover_s)],
        [Paragraph(ddr.get("property_address",""), sub_s)],
    ], colWidths=[175*mm], style=[
        ("BACKGROUND",(0,0),(-1,-1),C_DARK),
        ("TOPPADDING",(0,0),(0,0),20),("BOTTOMPADDING",(0,-1),(-1,-1),16),
        ("LEFTPADDING",(0,0),(-1,-1),12)
    ]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT))
    story.append(sp(10))

    for k,v in [("Customer", ddr.get("customer_name","Not Available")),
                ("Inspection Date", ddr.get("inspection_date","Not Available")),
                ("Report Generated", datetime.date.today().strftime("%d %B %Y"))]:
        story.append(Table([[Paragraph(k+":", label), Paragraph(v, body)]],
                           colWidths=[45*mm,130*mm],
                           style=[("TOPPADDING",(0,0),(-1,-1),4),
                                  ("BOTTOMPADDING",(0,0),(-1,-1),4),
                                  ("LEFTPADDING",(0,0),(-1,-1),6),
                                  ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDDDDD")),
                                  ("BACKGROUND",(0,0),(0,-1),C_LIGHT)]))
    story.append(sp(14))

    # ── Section 1: Summary ────────────────────────────────────────────────
    story.append(section_header("1", "Property Issue Summary"))
    story.append(sp(8))
    story.append(Paragraph(ddr.get("property_summary","Not Available"), body))
    story.append(sp(14))

    # ── Section 2: Area Observations ─────────────────────────────────────
    story.append(section_header("2", "Area-wise Observations"))
    story.append(sp(8))

    obs_list = ddr.get("observations", [])
    for i, obs in enumerate(obs_list):
        block = []
        block.append(Paragraph(obs.get("area",""), atitle))
        block.append(hr())
        block.append(sp(4))
        block.append(Paragraph(f"<b>Damage observed:</b> {obs.get('damage_observed','Not Available')}", body))
        block.append(Paragraph(f"<b>Source:</b> {obs.get('source_identified','Not Available')}", body))

        thermal = obs.get("thermal_findings","Not Available")
        if thermal != "Not Available":
            block.append(thermal_box(f"Thermal — {thermal}"))
        else:
            block.append(Paragraph("Thermal findings: Not Available", small))

        # Safely map the exact images identified by the AI
        row_imgs = []
        photo_indices = obs.get("photo_indices", [])
        
        # Limit to the first 2 photos identified for layout purposes
        for photo_num in photo_indices[:2]: 
            try:
                # Subtract 1 because "Photo 1" is at array index 0
                array_idx = int(photo_num) - 1 
                if 0 <= array_idx < len(insp_imgs):
                    ri = add_image(insp_imgs[array_idx], 78*mm, 58*mm)
                    if ri: row_imgs.append(ri)
            except (ValueError, TypeError):
                pass # Ignore if AI returns something that isn't a number

        # Thermal image: pick 1 per area slot
        therm_img = None
        if i < len(therm_imgs):
            ti = add_image(therm_imgs[i], 78*mm, 58*mm)
            if ti: therm_img = ti

        # Build image row
        if row_imgs or therm_img:
            cells = row_imgs[:2]
            if therm_img:
                cells.append(therm_img)
            while len(cells) < 3:
                cells.append(Paragraph("Image Not Available", small))
            block.append(sp(4))
            block.append(Table([cells], colWidths=[58*mm,58*mm,58*mm],
                               style=[("ALIGN",(0,0),(-1,-1),"CENTER"),
                                      ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                      ("TOPPADDING",(0,0),(-1,-1),4),
                                      ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
            captions = []
            captions.append(Paragraph("Inspection photo 1", small))
            captions.append(Paragraph("Inspection photo 2", small))
            captions.append(Paragraph("Thermal scan", small))
            block.append(Table([captions], colWidths=[58*mm,58*mm,58*mm],
                               style=[("ALIGN",(0,0),(-1,-1),"CENTER"),
                                      ("TOPPADDING",(0,0),(-1,-1),1)]))

        # Severity badge inline
        block.append(sp(4))
        block.append(Table([[Paragraph("Severity:", label),
                             sev_badge(obs.get("severity","Medium"))]],
                           colWidths=[25*mm,25*mm],
                           style=[("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                  ("TOPPADDING",(0,0),(-1,-1),4)]))
        block.append(sp(12))
        story.extend(block)

    story.append(sp(6))

    # ── Section 3: Root Causes ────────────────────────────────────────────
    story.append(section_header("3", "Probable Root Causes"))
    story.append(sp(8))
    for j, rc in enumerate(ddr.get("root_causes",[])):
        bg = C_LIGHT if j % 2 == 0 else WHITE
        story.append(Table([[
            Paragraph(f"{j+1}.", ParagraphStyle("n",fontName="Helvetica-Bold",
                      fontSize=12,textColor=C_ACCENT,leading=16)),
            [Paragraph(rc.get("issue",""), atitle),
             Paragraph(rc.get("cause",""), body)]
        ]], colWidths=[10*mm,160*mm], style=[
            ("BACKGROUND",(0,0),(-1,-1),bg),("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(0,-1),6),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDDDDD"))
        ]))
        story.append(sp(2))
    story.append(sp(12))

    # ── Section 4: Severity ───────────────────────────────────────────────
    story.append(section_header("4", "Severity Assessment"))
    story.append(sp(8))
    story.append(Paragraph(ddr.get("severity_summary","Not Available"), body))
    story.append(sp(6))

    # Per-area severity table
    rows = [["Area","Severity","Findings"]]
    for obs in obs_list:
        rows.append([obs.get("area",""),
                     obs.get("severity",""),
                     obs.get("damage_observed","")[:80]])
    col_w = [50*mm, 25*mm, 95*mm]
    t_style = [
        ("BACKGROUND",(0,0),(-1,0),C_DARK),("TEXTCOLOR",(0,0),(-1,0),WHITE),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8.5),
        ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDDDDD")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,C_LIGHT]),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]
    for ri, obs in enumerate(obs_list, 1):
        bg = SEV_COLOR.get(obs.get("severity",""), C_GRAY)
        t_style += [("BACKGROUND",(1,ri),(1,ri),bg),
                    ("TEXTCOLOR",(1,ri),(1,ri),WHITE),
                    ("FONTNAME",(1,ri),(1,ri),"Helvetica-Bold"),
                    ("ALIGN",(1,ri),(1,ri),"CENTER")]

    story.append(Table([[Paragraph(c, ParagraphStyle("th",fontName="Helvetica-Bold" if r==0 else "Helvetica",
                         fontSize=8.5,textColor=WHITE if r==0 else C_DARK,leading=13))
                        for c in row]
                       for r,row in enumerate(rows)],
                       colWidths=col_w, style=t_style))
    story.append(sp(14))

    # ── Section 5: Actions ────────────────────────────────────────────────
    story.append(section_header("5", "Recommended Actions"))
    story.append(sp(8))
    PCOL = {"Immediate": C_RED, "Short-term": C_ORANGE, "Long-term": C_GREEN}
    for act in ddr.get("actions",[]):
        p   = act.get("priority","Short-term")
        col = PCOL.get(p, C_GRAY)
        story.append(Table([[
            Paragraph(f"<b>{p}</b>", ParagraphStyle("pp",fontName="Helvetica-Bold",
                      fontSize=8,textColor=WHITE,leading=10,alignment=TA_CENTER)),
            Paragraph(f"{act.get('area','')} — {act.get('task','')}", body)
        ]], colWidths=[22*mm,148*mm], style=[
            ("BACKGROUND",(0,0),(0,0),col),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(0,0),4),("LEFTPADDING",(1,0),(1,0),8),
            ("LINEBELOW",(0,0),(-1,-1),0.3,colors.HexColor("#DDDDDD"))
        ]))
    story.append(sp(14))

    # ── Section 6: Notes ──────────────────────────────────────────────────
    story.append(section_header("6", "Additional Notes"))
    story.append(sp(8))
    story.append(Paragraph(ddr.get("additional_notes","None."), body))
    story.append(sp(14))

    # ── Section 7: Missing Info ───────────────────────────────────────────
    story.append(section_header("7", "Missing or Unclear Information"))
    story.append(sp(8))
    missing = ddr.get("missing_information",[])
    if not missing:
        story.append(Paragraph("None identified.", body))
    else:
        for m in missing:
            story.append(Table([[
                Paragraph("Not Available", ParagraphStyle("na",fontName="Helvetica-Bold",
                          fontSize=8,textColor=C_RED,leading=12)),
                Paragraph(str(m), small)
            ]], colWidths=[30*mm,140*mm], style=[
                ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#DDDDDD")),
                ("BACKGROUND",(0,0),(0,-1),C_LIGHT),
                ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                ("LEFTPADDING",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"MIDDLE")
            ]))
            story.append(sp(2))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(sp(12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_ACCENT))
    story.append(sp(4))
    story.append(Paragraph(
        f"AI-generated DDR | Findings sourced from provided documents only | "
        f"{datetime.date.today().strftime('%d %B %Y')}", foot))

    doc.build(story)
    print(f"  Done → {output}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--thermal",    required=True)
    parser.add_argument("--output",     default="Final_DDR.pdf")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("Error: set GROQ_API_KEY in your .env file")

    print("Step 1/4  Extracting text...")
    insp_text    = extract_text(args.inspection)
    therm_text   = extract_text(args.thermal)          # backup
    therm_reads  = extract_thermal_readings(args.thermal)
    thermal_summary = "\n".join(
        f"Scan {r['scan']}: Hotspot={r['hotspot']}C  Coldspot={r['coldspot']}C  "
        f"Delta-T={r['delta_t']}C  Emissivity={r['emissivity']}  Date={r['date']}"
        for r in therm_reads
    )
    print(f"  {len(insp_text):,} chars inspection | {len(therm_reads)} thermal readings")

    print("Step 2/4  Extracting images...")
    insp_imgs  = get_good_images(args.inspection, "insp", min_px=300)
    therm_imgs = get_good_images(args.thermal,    "therm", min_px=300)
    # Thermal PDF has 2 images per page (visual + IR scan), keep every other one (IR scan)
    therm_imgs = therm_imgs[1::2]   # index 1, 3, 5... = the IR coloured scans
    print(f"  {len(insp_imgs)} inspection photos | {len(therm_imgs)} thermal scans")

    print("Step 3/4  Calling Groq AI...")
    ddr = call_ai(insp_text, thermal_summary, api_key)
    with open(args.output.replace(".pdf","_raw.json"), "w") as f:
        json.dump(ddr, f, indent=2)
    print(f"  {len(ddr.get('observations',[]))} areas extracted")

    print("Step 4/4  Rendering PDF...")
    render(ddr, insp_imgs, therm_imgs, args.output)


if __name__ == "__main__":
    main()