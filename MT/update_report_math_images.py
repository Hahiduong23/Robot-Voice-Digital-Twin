import re

with open("ADL/create_word_report.py", "r") as f:
    code = f.read()

# Replace add_native_equation function definition with add_equation_image
old_func = '''def add_native_equation(doc, omml_body, eq_label):
    """
    Inserts a native Microsoft Word Equation using Office Math (m:oMath)
    in a 1-row, 2-col borderless table for centered math and right-aligned label.
    """
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="FFFFFF", sz="0", val="none")

    c_eq, c_num = tbl.rows[0].cells
    c_eq.width = Inches(5.8)
    c_num.width = Inches(0.8)
    set_cell_margins(c_eq, top=40, bottom=40, left=40, right=40)
    set_cell_margins(c_num, top=40, bottom=40, left=40, right=40)

    p_eq = c_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(3)
    p_eq.paragraph_format.space_after = Pt(3)
    
    omath_xml = f'<m:oMath {NS_MATH}>{omml_body}</m:oMath>'
    p_eq._p.append(parse_xml(omath_xml))

    p_num = c_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(3)
    p_num.paragraph_format.space_after = Pt(3)
    r_num = p_num.add_run(eq_label)
    r_num.font.name = 'Times New Roman'
    r_num.font.size = Pt(11)
    r_num.font.color.rgb = RGBColor(100, 100, 100)'''

new_func = '''from PIL import Image

def add_equation_image(doc, img_path, eq_label, max_width_in=5.4):
    """
    Inserts high-resolution LaTeX equation rendered as an image in a 1-row, 2-col borderless table.
    Left cell: Centered equation image.
    Right cell: Right-aligned equation number.
    """
    tbl = doc.add_table(rows=1, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl, color="FFFFFF", sz="0", val="none")

    c_eq, c_num = tbl.rows[0].cells
    c_eq.width = Inches(5.8)
    c_num.width = Inches(0.8)
    set_cell_margins(c_eq, top=50, bottom=50, left=30, right=30)
    set_cell_margins(c_num, top=50, bottom=50, left=30, right=30)

    p_eq = c_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(2)
    p_eq.paragraph_format.space_after = Pt(2)

    im = Image.open(img_path)
    w_px, h_px = im.size
    aspect = w_px / h_px
    target_h = 0.55
    calc_w = min(max_width_in, max(2.4, target_h * aspect))

    p_eq.add_run().add_picture(img_path, width=Inches(calc_w))

    p_num = c_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(10)
    p_num.paragraph_format.space_after = Pt(2)
    r_num = p_num.add_run(eq_label)
    r_num.font.name = 'Times New Roman'
    r_num.font.size = Pt(11)
    r_num.font.bold = True
    r_num.font.color.rgb = RGBColor(16, 44, 87)'''

code = code.replace(old_func, new_func)

# Replace all equation calls
eq_mappings = [
    ('add_native_equation(doc, EQ1_OMML, "(1)")', 'add_equation_image(doc, "ADL/results/equations/eq1_center_pad.png", "(1)")'),
    ('add_native_equation(doc, EQ2_OMML, "(2)")', 'add_equation_image(doc, "ADL/results/equations/eq2_stft.png", "(2)")'),
    ('add_native_equation(doc, EQ3_OMML, "(3)")', 'add_equation_image(doc, "ADL/results/equations/eq3_mel.png", "(3)")'),
    ('add_native_equation(doc, EQ4_OMML, "(4)")', 'add_equation_image(doc, "ADL/results/equations/eq4_batchnorm.png", "(4)")'),
    ('add_native_equation(doc, EQ5_OMML, "(5)")', 'add_equation_image(doc, "ADL/results/equations/eq5_residual.png", "(5)")'),
    ('add_native_equation(doc, EQ6_OMML, "(6)")', 'add_equation_image(doc, "ADL/results/equations/eq6_squeeze.png", "(6)")'),
    ('add_native_equation(doc, EQ7_OMML, "(7)")', 'add_equation_image(doc, "ADL/results/equations/eq7_excitation.png", "(7)")'),
    ('add_native_equation(doc, EQ8_OMML, "(8)")', 'add_equation_image(doc, "ADL/results/equations/eq8_cross_entropy.png", "(8)")'),
    ('add_native_equation(doc, EQ9_OMML, "(9)")', 'add_equation_image(doc, "ADL/results/equations/eq9_label_smoothing.png", "(9)")'),
    ('add_native_equation(doc, EQ10_OMML, "(10)")', 'add_equation_image(doc, "ADL/results/equations/eq10_cosine_annealing.png", "(10)")'),
    ('add_native_equation(doc, EQ11_OMML, "(11)")', 'add_equation_image(doc, "ADL/results/equations/eq11_gradcam_alpha.png", "(11)")'),
    ('add_native_equation(doc, EQ12_OMML, "(12)")', 'add_equation_image(doc, "ADL/results/equations/eq12_gradcam_map.png", "(12)")'),
]

for old_call, new_call in eq_mappings:
    code = code.replace(old_call, new_call)

with open("ADL/create_word_report.py", "w") as f:
    f.write(code)

print("[SUCCESS] create_word_report.py updated with equation image embeddings!")
