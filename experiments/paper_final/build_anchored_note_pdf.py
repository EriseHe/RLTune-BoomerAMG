"""Render the reviewed schedule derivation as a compact four-page PDF."""

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

from pathlib import Path
import argparse
import io
import subprocess


def build(path,math_python=None):
    if math_python is None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.mathtext import math_to_image
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak,Table,TableStyle
    from pypdf import PdfReader
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    W,H=A4;content=W-96;story=[];buffers=[]
    body=ParagraphStyle("body",fontName="Helvetica",fontSize=10.2,leading=15,spaceAfter=9)
    title=ParagraphStyle("title",fontName="Helvetica-Bold",fontSize=21,leading=26,spaceAfter=14,textColor=colors.HexColor("#173c52"))
    head=ParagraphStyle("head",parent=body,fontName="Helvetica-Bold",fontSize=12.5,leading=17,spaceBefore=10,spaceAfter=8,textColor=colors.HexColor("#173c52"))
    small=ParagraphStyle("small",parent=body,fontSize=8.8,leading=12)
    def p(text,style=body):story.append(Paragraph(text,style))
    def eq(text):
        buf=io.BytesIO()
        if math_python is None:
            math_to_image("$"+text+"$",buf,dpi=240,format="png",color="#102a3a")
        else:
            command="import sys,matplotlib;matplotlib.use('Agg');from matplotlib.mathtext import math_to_image;math_to_image(sys.stdin.read(),sys.stdout.buffer,dpi=240,format='png',color='#102a3a')"
            result=subprocess.run([str(math_python),"-c",command],input=("$"+text+"$").encode(),capture_output=True,check=True)
            buf.write(result.stdout)
        buf.seek(0);im=PILImage.open(buf);w,h=im.size;scale=min(content/w,72/240)
        story.append(Image(buf,width=w*scale,height=h*scale,hAlign="CENTER"));story.append(Spacer(1,10));buffers.append(buf)
    def page():story.append(PageBreak())
    p("A prescribed anchored schedule",title)
    p("Reviewed mathematical basis for (2.6, 1) | Module 05 Run 04",head)
    p("The coefficient is selected analytically for an SPD polynomial-smoothing surrogate. The complete multilevel BoomerAMG comparison remains an empirical test. The original proof was reviewed against the actual smoother before Run 04 began.")
    p("1. Normalize the smoother",head)
    eq(r"D_{ii}=\sum_j|a_{ij}|,\quad 0\prec A\preceq D,\quad T=A^{1/2}D^{-1}A^{1/2}")
    p("For SPD A, pair its symmetric off-diagonal terms and use 2|x<sub>i</sub>x<sub>j</sub>| &le; x<sub>i</sub><super>2</super>+x<sub>j</sub><super>2</super>. This proves A &le; D as quadratic forms. T is symmetric positive definite and has eigenvalues in (0,1]. A weighted sweep in energy coordinates is S(w)=I-wT.")
    p("2. Fix one weight at unity",head)
    eq(r"p_a(t)=(1-t)(1-at),\qquad \eta(a)=\max_{0\leq t\leq1}t(1-t)^2(1-at)^2")
    p("The weighted objective is the square of the smoothing quantity in Lottes's two-level bound (equation 4). Anchoring one step at the nominal weight 1 is an explicit design restriction. The weights' sum and product define the error polynomial; their difference is not another effective relaxation weight.")
    eq(r"(I-bT)(I-aT)=I-(a+b)T+abT^2")
    p("Result",head)
    eq(r"a_*=\frac{3+\sqrt{5}}{2}=2.6180339887\ldots,\qquad\eta(a_*)=\frac{10-2\sqrt{5}}{125}")
    p("On the exact action grid {1,1.05,...,3}, the minimizer is <b>2.6</b>. This selects the pair (2.6,1) before observing Run 04 timings.")
    page()
    p("The minimax proof",title)
    eq(r"f_a(t)=\sqrt{t}(1-t)(1-at),\qquad f'_a(t)=\frac{1-3(a+1)t+5at^2}{2\sqrt{t}}")
    p("At a=a*, the only two interior stationary points are")
    eq(r"t_+=1-\frac{2}{\sqrt{5}},\qquad t_-=\frac{1}{2}\left(1+\frac{1}{\sqrt{5}}\right)")
    p("Direct substitution gives equal positive and negative peaks:")
    eq(r"f_{a_*}(t_+)=g,\quad f_{a_*}(t_-)=-g,\qquad g^2=\frac{10-2\sqrt{5}}{125}")
    p("Both endpoints vanish, so these peaks determine the maximum. Evaluating any alternative coefficient at the same two points gives")
    eq(r"a<a_*\ \Longrightarrow\ f_a(t_+)=g+(a_*-a)t_+^{3/2}(1-t_+)>g")
    eq(r"a>a_*\ \Longrightarrow\ f_a(t_-)=-g-(a-a_*)t_-^{3/2}(1-t_-)<-g")
    p("Every other real coefficient therefore has a strictly larger maximum absolute value. Squaring proves global optimality and uniqueness for this objective.")
    p("Discrete evaluation at analytic extrema",head)
    rows=[["High coefficient a","eta(a)"],["2.50","0.0457988174103"],["2.60","0.0444568224570"],["a* = 2.61803398875...","0.0442229123600"],["2.65","0.0465429476041"]]
    table=Table(rows,colWidths=[content*.57,content*.43],rowHeights=25)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#173c52")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("BACKGROUND",(0,2),(-1,2),colors.HexColor("#eaf2f6")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBELOW",(0,-1),(-1,-1),.5,colors.lightgrey)]));story.append(table);story.append(Spacer(1,12))
    p("All 41 grid values were evaluated at stationary points, rather than estimated from a sampled spectral plot or nearest rounding. The differences above are surrogate differences, not predicted runtime savings.",small)
    page()
    p("The exact two-grid connection",title)
    p("Assume a fixed exact orthogonal coarse-correction projection Q in energy coordinates, and one pre-sweep plus one post-sweep per cycle:")
    eq(r"Q=Q^T=Q^2,\quad E(w)=(I-wT)Q(I-wT),\quad P_a=p_a(T)")
    eq(r"\rho(E(1)E(a))=\|QP_aQ\|_2^2\leq K\eta(a),\qquad K=\|T^{-1/2}Q\|_2^2")
    p("Proof",head)
    p("Write S<sub>1</sub>=I-T and S<sub>a</sub>=I-aT. They commute. The two-cycle product is S<sub>1</sub>QP<sub>a</sub>QS<sub>a</sub>. Cyclically moving the last factor to the front preserves its characteristic polynomial, giving (P<sub>a</sub>Q)<super>2</super>.")
    p("In a basis for range(Q) and ker(Q), P<sub>a</sub>Q is block triangular with a symmetric compression on range(Q) and zeros on the remaining diagonal. Its spectral radius is therefore the norm of QP<sub>a</sub>Q. Squaring gives the identity. The bound follows from")
    eq(r"\|QP_aQ\|_2\leq\|P_aQ\|_2\leq\|P_aT^{1/2}\|_2\,\|T^{-1/2}Q\|_2\leq\sqrt{\eta(a)K}")
    p("This argument is valid even when a smoothing factor is singular. It does not equate the norm of the original nonnormal product with its spectral radius. Reversing the two cycles gives the same eigenvalues, but can change intermediate residuals and stopping times.")
    p("What the bound does and does not establish",head)
    p("For fixed K, minimizing eta minimizes this upper bound. K may be large and the bound can exceed one. The result gives neither a universal finite-tolerance time advantage nor a best starting phase. A high step may amplify some modes before the paired step damps them.")
    p("In a multilevel V-cycle the recursive coarse solver depends on the chosen weight; it need not be a fixed exact projection. A direct solve only on the last level does not remove this distinction. Nonsymmetric advection also falls outside the SPD assumptions.")
    page()
    p("Implementation and paper use",title)
    p("The implemented smoother matches the normalization",head)
    p("The profile <b>l1_jacobi_direct_coarse</b> uses type 18 on down/up stages and type 9 on the coarsest stage. Natural ordering is the native default and no frozen job overrides it. Type 18 therefore uses the full absolute row sum, and the old-iterate Jacobi kernel applies x &larr; x + wD<super>-1</super>(b-Ax).")
    p("Every action applies one weight to a complete V-cycle with one pre-sweep and one post-sweep. The weight is set on all smoothing levels. All 600 test jobs have positive diffusion coefficients and zero advection. Native checks confirm the 18/18/9 profile and V-cycle for all six checkpoints.")
    p("The actual multilevel limitation",head)
    eq(r"E_0(w)=S_0(w)[I-P_0B_1(w)P_0^TA_0]S_0(w),\quad S_0(w)=I-wD_0^{-1}A_0")
    p("Here the recursive coarse solver B<sub>1</sub>(w) generally varies with w. The exact two-grid theorem is a justified model for coefficient design, while production runtime is measured. Coarse-level SPD preservation also assumes a full-rank Galerkin interpolation in exact arithmetic.")
    p("Suggested paper wording",head)
    p("We include a prescribed period-two relaxation baseline with weights (2.6,1), motivated by normalized SPD l1-Jacobi polynomial smoothing. Anchoring one coefficient at unity and minimizing the weighted smoothing surrogate gives a*=(3+sqrt(5))/2; analytic evaluation on our 0.05 grid selects a=2.6. An exact two-grid model links this surrogate to an upper bound on the two-cycle spectral radius. We transfer the coefficients to complete multilevel cycles and assess their finite-tolerance cost empirically.")
    p("Verification and protocol",head)
    p("The companion script verifies the symbolic extrema, 41 grid coefficients, 500 matrix identities/bounds (maximum identity error about 1.2e-15), all 600 input jobs, and six native profiles. Run 04 keeps Run 03's checkpoints, inputs, fixed weights, three repetitions, tolerance, cycle cap and recovery; only (2.5,1) becomes (2.6,1). High-first phase is prescribed and resets each primary solve.",small)
    p('References: J. Lottes, <i>Optimal polynomial smoothers for multigrid V-cycles</i>, 2023, <link href="https://arxiv.org/pdf/2202.08830v3" color="#145f88">arXiv:2202.08830v3</link>, equations (4), (7), section 3.2. P. D\'Ambra et al., <i>Optimal Polynomial Smoothers for Parallel AMG</i>, <link href="https://arxiv.org/html/2407.09848v3" color="#145f88">arXiv:2407.09848v3</link>, equation (7). The anchored specialization is proved here; no priority claim is made.',small)
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#657785"))
        canvas.drawString(48,28,"Anchored schedule | reviewed 27 September 2026")
        canvas.drawRightString(W-48,28,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=47,
        title="Reviewed anchored schedule: mathematical basis for (2.6,1)").build(story,onFirstPage=footer,onLaterPages=footer)
    pages=len(PdfReader(path).pages)
    if pages!=4:raise AssertionError(f"Expected four intentional pages; got {pages}")
    return path


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    p.add_argument("--math-python",type=Path)
    args=p.parse_args();print(build(args.output,args.math_python))
