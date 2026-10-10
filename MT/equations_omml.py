import docx
from docx.oxml import parse_xml

# Helper namespace definition
NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'

def wrap_omml(omml_content, eq_num=""):
    """
    Wraps OMML content in oMathPara. If eq_num is provided, includes right tab and equation number.
    """
    if eq_num:
        return f'''
        <m:oMathPara {NS}>
          <m:oMath>
            {omml_content}
          </m:oMath>
          <w:r><w:tab/><w:tab/><w:tab/><w:t>{eq_num}</w:t></w:r>
        </m:oMathPara>
        '''
    else:
        return f'''
        <m:oMathPara {NS}>
          <m:oMath>
            {omml_content}
          </m:oMath>
        </m:oMathPara>
        '''

# EQ 1: Center-Pad/Crop
EQ1_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSub><m:e><m:r><m:t>x</m:t></m:r></m:e><m:sub><m:r><m:t>pad</m:t></m:r></m:sub></m:sSub>
    <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e><m:r><m:t>n</m:t></m:r></m:e></m:d>
    <m:r><m:t> = </m:t></m:r>
    <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e>
      <m:sSub><m:e><m:r><m:t>0</m:t></m:r></m:e><m:sub><m:r><m:t>left</m:t></m:r></m:sub></m:sSub>
      <m:r><m:t>,  x</m:t></m:r>
      <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e><m:r><m:t>n</m:t></m:r></m:e></m:d>
      <m:r><m:t>,  </m:t></m:r>
      <m:sSub><m:e><m:r><m:t>0</m:t></m:r></m:e><m:sub><m:r><m:t>right</m:t></m:r></m:sub></m:sSub>
    </m:e></m:d>
    <m:r><m:t>,      </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>pad</m:t></m:r></m:e><m:sub><m:r><m:t>left</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> = </m:t></m:r>
    <m:d><m:dPr><m:begChr m:val="⌊"/><m:endChr m:val="⌋"/></m:dPr><m:e>
      <m:f>
        <m:num>
          <m:sSub><m:e><m:r><m:t>N</m:t></m:r></m:e><m:sub><m:r><m:t>target</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t> − L</m:t></m:r>
        </m:num>
        <m:den><m:r><m:t>2</m:t></m:r></m:den>
      </m:f>
    </m:e></m:d>
    <m:r><m:t>,      </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>pad</m:t></m:r></m:e><m:sub><m:r><m:t>right</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> = </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>N</m:t></m:r></m:e><m:sub><m:r><m:t>target</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> − L − </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>pad</m:t></m:r></m:e><m:sub><m:r><m:t>left</m:t></m:r></m:sub></m:sSub>
  </m:oMath>
</m:oMathPara>
'''

# EQ 2: STFT
EQ2_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:r><m:t>X(m, ω) = </m:t></m:r>
    <m:nary>
      <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
      <m:sub><m:r><m:t>n=−∞</m:t></m:r></m:sub>
      <m:sup><m:r><m:t>∞</m:t></m:r></m:sup>
      <m:e>
        <m:r><m:t>x[n] · w[n − m · H] · </m:t></m:r>
        <m:sSup>
          <m:e><m:r><m:t>e</m:t></m:r></m:e>
          <m:sup><m:r><m:t>−j · ω · n</m:t></m:r></m:sup>
        </m:sSup>
      </m:e>
    </m:nary>
  </m:oMath>
</m:oMathPara>
'''

# EQ 3: Mel-Spectrogram
EQ3_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSub><m:e><m:r><m:t>S</m:t></m:r></m:e><m:sub><m:r><m:t>mel</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>(m, </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>f</m:t></m:r></m:e><m:sub><m:r><m:t>mel</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>) = 10 · </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>log</m:t></m:r></m:e><m:sub><m:r><m:t>10</m:t></m:r></m:sub></m:sSub>
    <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e>
      <m:nary>
        <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
        <m:sub><m:r><m:t>k=0</m:t></m:r></m:sub>
        <m:sup><m:f><m:num><m:sSub><m:e><m:r><m:t>N</m:t></m:r></m:e><m:sub><m:r><m:t>fft</m:t></m:r></m:sub></m:sSub></m:num><m:den><m:r><m:t>2</m:t></m:r></m:den></m:f></m:sup>
        <m:e>
          <m:sSup><m:e><m:r><m:t>|X(m, k)|</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>
          <m:r><m:t> · </m:t></m:r>
          <m:sSub><m:e><m:r><m:t>H</m:t></m:r></m:e><m:sub><m:r><m:t>mel</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t>(k, </m:t></m:r>
          <m:sSub><m:e><m:r><m:t>f</m:t></m:r></m:e><m:sub><m:r><m:t>mel</m:t></m:r></m:sub></m:sSub>
          <m:r><m:t>) + ε</m:t></m:r>
        </m:e>
      </m:nary>
    </m:e></m:d>
  </m:oMath>
</m:oMathPara>
'''

# EQ 4: BatchNorm
EQ4_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSup>
      <m:e><m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e><m:r><m:t>x</m:t></m:r></m:e></m:acc></m:e>
      <m:sup><m:r><m:t>(k)</m:t></m:r></m:sup>
    </m:sSup>
    <m:r><m:t> = </m:t></m:r>
    <m:f>
      <m:num>
        <m:sSup><m:e><m:r><m:t>x</m:t></m:r></m:e><m:sup><m:r><m:t>(k)</m:t></m:r></m:sup></m:sSup>
        <m:r><m:t> − </m:t></m:r>
        <m:sSub><m:e><m:r><m:t>μ</m:t></m:r></m:e><m:sub><m:r><m:t>B</m:t></m:r></m:sub></m:sSub>
      </m:num>
      <m:den>
        <m:rad>
          <m:radPr><m:degHide m:val="on"/></m:radPr><m:deg/>
          <m:e>
            <m:sSubSup>
              <m:e><m:r><m:t>σ</m:t></m:r></m:e>
              <m:sub><m:r><m:t>B</m:t></m:r></m:sub>
              <m:sup><m:r><m:t>2</m:t></m:r></m:sup>
            </m:sSubSup>
            <m:r><m:t> + ε</m:t></m:r>
          </m:e>
        </m:rad>
      </m:den>
    </m:f>
    <m:r><m:t>,        </m:t></m:r>
    <m:sSup><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sup><m:r><m:t>(k)</m:t></m:r></m:sup></m:sSup>
    <m:r><m:t> = </m:t></m:r>
    <m:sSup><m:e><m:r><m:t>γ</m:t></m:r></m:e><m:sup><m:r><m:t>(k)</m:t></m:r></m:sup></m:sSup>
    <m:sSup>
      <m:e><m:acc><m:accPr><m:chr m:val="̂"/></m:accPr><m:e><m:r><m:t>x</m:t></m:r></m:e></m:acc></m:e>
      <m:sup><m:r><m:t>(k)</m:t></m:r></m:sup>
    </m:sSup>
    <m:r><m:t> + </m:t></m:r>
    <m:sSup><m:e><m:r><m:t>β</m:t></m:r></m:e><m:sup><m:r><m:t>(k)</m:t></m:r></m:sup></m:sSup>
  </m:oMath>
</m:oMathPara>
'''

# EQ 5: Residual
EQ5_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:r><m:t>y = </m:t></m:r>
    <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>F</m:t></m:r>
    <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
      <m:r><m:t>x, </m:t></m:r>
      <m:d><m:dPr><m:begChr m:val="{"/><m:endChr m:val="}"/></m:dPr><m:e>
        <m:sSub><m:e><m:r><m:t>W</m:t></m:r></m:e><m:sub><m:r><m:t>i</m:t></m:r></m:sub></m:sSub>
      </m:e></m:d>
    </m:e></m:d>
    <m:r><m:t> + x,      where  dim(</m:t></m:r>
    <m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>F</m:t></m:r>
    <m:r><m:t>) = dim(x)</m:t></m:r>
  </m:oMath>
</m:oMathPara>
'''

# EQ 6: Squeeze
EQ6_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSub><m:e><m:r><m:t>z</m:t></m:r></m:e><m:sub><m:r><m:t>c</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> = </m:t></m:r>
    <m:sSub>
      <m:e><m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>F</m:t></m:r></m:e>
      <m:sub><m:r><m:t>sq</m:t></m:r></m:sub>
    </m:sSub>
    <m:r><m:t>(</m:t></m:r>
    <m:sSub><m:e><m:r><m:t>u</m:t></m:r></m:e><m:sub><m:r><m:t>c</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>) = </m:t></m:r>
    <m:f>
      <m:num><m:r><m:t>1</m:t></m:r></m:num>
      <m:den><m:r><m:t>H × W</m:t></m:r></m:den>
    </m:f>
    <m:nary>
      <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
      <m:sub><m:r><m:t>i=1</m:t></m:r></m:sub>
      <m:sup><m:r><m:t>H</m:t></m:r></m:sup>
      <m:e>
        <m:nary>
          <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
          <m:sub><m:r><m:t>j=1</m:t></m:r></m:sub>
          <m:sup><m:r><m:t>W</m:t></m:r></m:sup>
          <m:e>
            <m:sSub><m:e><m:r><m:t>u</m:t></m:r></m:e><m:sub><m:r><m:t>c</m:t></m:r></m:sub></m:sSub>
            <m:r><m:t>(i, j)</m:t></m:r>
          </m:e>
        </m:nary>
      </m:e>
    </m:nary>
  </m:oMath>
</m:oMathPara>
'''

# EQ 7: Excitation
EQ7_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:r><m:t>s = </m:t></m:r>
    <m:sSub>
      <m:e><m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>F</m:t></m:r></m:e>
      <m:sub><m:r><m:t>ex</m:t></m:r></m:sub>
    </m:sSub>
    <m:r><m:t>(z, W) = σ(</m:t></m:r>
    <m:sSub><m:e><m:r><m:t>W</m:t></m:r></m:e><m:sub><m:r><m:t>2</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> · ReLU(</m:t></m:r>
    <m:sSub><m:e><m:r><m:t>W</m:t></m:r></m:e><m:sub><m:r><m:t>1</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> · z)),        </m:t></m:r>
    <m:sSub>
      <m:e><m:acc><m:accPr><m:chr m:val="̃"/></m:accPr><m:e><m:r><m:t>x</m:t></m:r></m:e></m:acc></m:e>
      <m:sub><m:r><m:t>c</m:t></m:r></m:sub>
    </m:sSub>
    <m:r><m:t> = </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>s</m:t></m:r></m:e><m:sub><m:r><m:t>c</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> · </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>u</m:t></m:r></m:e><m:sub><m:r><m:t>c</m:t></m:r></m:sub></m:sSub>
  </m:oMath>
</m:oMathPara>
'''

# EQ 8: Cross-Entropy
EQ8_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSub>
      <m:e><m:r><m:rPr><m:scr m:val="script"/></m:rPr><m:t>L</m:t></m:r></m:e>
      <m:sub><m:r><m:t>CE</m:t></m:r></m:sub>
    </m:sSub>
    <m:r><m:t> = − </m:t></m:r>
    <m:nary>
      <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
      <m:sub><m:r><m:t>k=1</m:t></m:r></m:sub>
      <m:sup><m:r><m:t>K</m:t></m:r></m:sup>
      <m:e>
        <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>k</m:t></m:r></m:sub></m:sSub>
        <m:r><m:t> · log(</m:t></m:r>
        <m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>k</m:t></m:r></m:sub></m:sSub>
        <m:r><m:t>),        where    </m:t></m:r>
        <m:sSub><m:e><m:r><m:t>p</m:t></m:r></m:e><m:sub><m:r><m:t>k</m:t></m:r></m:sub></m:sSub>
        <m:r><m:t> = </m:t></m:r>
        <m:f>
          <m:num>
            <m:sSup><m:e><m:r><m:t>e</m:t></m:r></m:e><m:sup><m:sSub><m:e><m:r><m:t>z</m:t></m:r></m:e><m:sub><m:r><m:t>k</m:t></m:r></m:sub></m:sSub></m:sup></m:sSup>
          </m:num>
          <m:den>
            <m:nary>
              <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
              <m:sub><m:r><m:t>j=1</m:t></m:r></m:sub>
              <m:sup><m:r><m:t>K</m:t></m:r></m:sup>
              <m:e>
                <m:sSup><m:e><m:r><m:t>e</m:t></m:r></m:e><m:sup><m:sSub><m:e><m:r><m:t>z</m:t></m:r></m:e><m:sub><m:r><m:t>j</m:t></m:r></m:sub></m:sSub></m:sup></m:sSup>
              </m:e>
            </m:nary>
          </m:den>
        </m:f>
      </m:e>
    </m:nary>
  </m:oMath>
</m:oMathPara>
'''

# EQ 9: Label Smoothing
EQ9_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSubSup>
      <m:e><m:r><m:t>y</m:t></m:r></m:e>
      <m:sub><m:r><m:t>k</m:t></m:r></m:sub>
      <m:sup><m:r><m:t>LS</m:t></m:r></m:sup>
    </m:sSubSup>
    <m:r><m:t> = (1 − α) · </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>k</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> + </m:t></m:r>
    <m:f><m:num><m:r><m:t>α</m:t></m:r></m:num><m:den><m:r><m:t>K</m:t></m:r></m:den></m:f>
    <m:r><m:t> = 0.9 · </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>y</m:t></m:r></m:e><m:sub><m:r><m:t>k</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> + </m:t></m:r>
    <m:f><m:num><m:r><m:t>0.1</m:t></m:r></m:num><m:den><m:r><m:t>7</m:t></m:r></m:den></m:f>
  </m:oMath>
</m:oMathPara>
'''

# EQ 10: Cosine Annealing
EQ10_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSub><m:e><m:r><m:t>η</m:t></m:r></m:e><m:sub><m:r><m:t>t</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> = </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>η</m:t></m:r></m:e><m:sub><m:r><m:t>min</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> + </m:t></m:r>
    <m:f><m:num><m:r><m:t>1</m:t></m:r></m:num><m:den><m:r><m:t>2</m:t></m:r></m:den></m:f>
    <m:r><m:t> (</m:t></m:r>
    <m:sSub><m:e><m:r><m:t>η</m:t></m:r></m:e><m:sub><m:r><m:t>max</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t> − </m:t></m:r>
    <m:sSub><m:e><m:r><m:t>η</m:t></m:r></m:e><m:sub><m:r><m:t>min</m:t></m:r></m:sub></m:sSub>
    <m:r><m:t>) · </m:t></m:r>
    <m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e>
      <m:r><m:t>1 + cos</m:t></m:r>
      <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
        <m:f>
          <m:num>
            <m:sSub><m:e><m:r><m:t>T</m:t></m:r></m:e><m:sub><m:r><m:t>cur</m:t></m:r></m:sub></m:sSub>
          </m:num>
          <m:den>
            <m:sSub><m:e><m:r><m:t>T</m:t></m:r></m:e><m:sub><m:r><m:t>max</m:t></m:r></m:sub></m:sSub>
          </m:den>
        </m:f>
        <m:r><m:t> · π</m:t></m:r>
      </m:e></m:d>
    </m:e></m:d>
  </m:oMath>
</m:oMathPara>
'''

# EQ 11: Grad-CAM alpha
EQ11_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSubSup>
      <m:e><m:r><m:t>α</m:t></m:r></m:e>
      <m:sub><m:r><m:t>k</m:t></m:r></m:sub>
      <m:sup><m:r><m:t>c</m:t></m:r></m:sup>
    </m:sSubSup>
    <m:r><m:t> = </m:t></m:r>
    <m:f><m:num><m:r><m:t>1</m:t></m:r></m:num><m:den><m:r><m:t>Z</m:t></m:r></m:den></m:f>
    <m:nary>
      <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
      <m:sub><m:r><m:t>i=1</m:t></m:r></m:sub>
      <m:sup><m:r><m:t>H</m:t></m:r></m:sup>
      <m:e>
        <m:nary>
          <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
          <m:sub><m:r><m:t>j=1</m:t></m:r></m:sub>
          <m:sup><m:r><m:t>W</m:t></m:r></m:sup>
          <m:e>
            <m:f>
              <m:num>
                <m:r><m:t>∂ </m:t></m:r>
                <m:sSup><m:e><m:r><m:t>Y</m:t></m:r></m:e><m:sup><m:r><m:t>c</m:t></m:r></m:sup></m:sSup>
              </m:num>
              <m:den>
                <m:r><m:t>∂ </m:t></m:r>
                <m:sSubSup>
                  <m:e><m:r><m:t>A</m:t></m:r></m:e>
                  <m:sub><m:r><m:t>i,j</m:t></m:r></m:sub>
                  <m:sup><m:r><m:t>k</m:t></m:r></m:sup>
                </m:sSubSup>
              </m:den>
            </m:f>
          </m:e>
        </m:nary>
      </m:e>
    </m:nary>
  </m:oMath>
</m:oMathPara>
'''

# EQ 12: Grad-CAM Map
EQ12_OMML = f'''
<m:oMathPara {NS}>
  <m:oMath>
    <m:sSubSup>
      <m:e><m:r><m:t>L</m:t></m:r></m:e>
      <m:sub><m:r><m:t>Grad-CAM</m:t></m:r></m:sub>
      <m:sup><m:r><m:t>c</m:t></m:r></m:sup>
    </m:sSubSup>
    <m:r><m:t> = ReLU</m:t></m:r>
    <m:d><m:dPr><m:begChr m:val="("/><m:endChr m:val=")"/></m:dPr><m:e>
      <m:nary>
        <m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/></m:naryPr>
        <m:sub><m:r><m:t>k</m:t></m:r></m:sub>
        <m:sup/>
        <m:e>
          <m:sSubSup>
            <m:e><m:r><m:t>α</m:t></m:r></m:e>
            <m:sub><m:r><m:t>k</m:t></m:r></m:sub>
            <m:sup><m:r><m:t>c</m:t></m:r></m:sup>
          </m:sSubSup>
          <m:sSup><m:e><m:r><m:t>A</m:t></m:r></m:e><m:sup><m:r><m:t>k</m:t></m:r></m:sup></m:sSup>
        </m:e>
      </m:nary>
    </m:e></m:d>
  </m:oMath>
</m:oMathPara>
'''

if __name__ == "__main__":
    doc = docx.Document()
    eqs = [EQ1_OMML, EQ2_OMML, EQ3_OMML, EQ4_OMML, EQ5_OMML, EQ6_OMML, EQ7_OMML, EQ8_OMML, EQ9_OMML, EQ10_OMML, EQ11_OMML, EQ12_OMML]
    for idx, eq in enumerate(eqs):
        p = doc.add_paragraph()
        p._p.append(parse_xml(eq))
    doc.save("ADL/test_all_omml.docx")
    print("[SUCCESS] All 12 native OMML equations validated and saved to test_all_omml.docx!")
