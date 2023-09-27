import pandas as pd


def build_emissionlinesdict():
    """  
    Build a dictionnary of lines in galaxies
    
    """
    
    
    #http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
    #https://classic.sdss.org/dr6/algorithms/linestable.html



    df_lines=pd.read_excel("datatools/GalEmissionLines.xlsx")
    df_sdss_lines = pd.read_excel("datatools/sdss_galaxylines.xlsx")
    
    lines_to_plot={}
    
    # K
    sel = df_sdss_lines["species"] == 'K'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["K"]={"wls":wls,"name":"K","type":"absorption"}
    
    # H
    sel = df_sdss_lines["species"] == 'H'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["H"]={"wls":wls,"name":"H","type":"absorption"}
    
    # G
    sel = df_sdss_lines["species"] == 'G'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["G"]={"wls":wls,"name":"G","type":"absorption"}
    
    # Mg
    sel = df_sdss_lines["species"] == 'Mg'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["Mg"]={"wls":wls,"name":"Mg","type":"absorption"}
    
    # Na
    sel = df_sdss_lines["species"] == 'Na'
    wls = df_sdss_lines[sel]["wl"].values
    lines_to_plot["Na"]={"wls":wls,"name":"Na","type":"absorption"}
    
    # H8
    sel = df_lines["ion"] == 'H8'
    wls = df_lines[sel]["wl"].values
    lines_to_plot["H8"]={"wls":wls,"name":"$H8$","type":"emission"}
    
    # H9
    sel = df_lines["ion"] == 'H9'
    wls = df_lines[sel]["wl"].values
    lines_to_plot["H9"]={"wls":wls,"name":"$H9$","type":"emission"}
    
    # H10
    sel = df_lines["ion"] == 'H10'
    wls = df_lines[sel]["wl"].values
    lines_to_plot["H10"]={"wls":wls,"name":"$H10$","type":"emission"}
    
    # H11
    sel = df_lines["ion"] == 'H11'
    wls = df_lines[sel]["wl"].values
    lines_to_plot["H11"]={"wls":wls,"name":"$H11$","type":"emission"}
    
    # Halpha
    sel = df_lines["ion"] == 'Hα' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{alpha}"]={"wls":wls,"name":"$H_\\alpha$","type":"emission"}
    
    
    # Hbeta
    sel = df_lines["ion"] == 'Hβ' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{beta}"]={"wls":wls,"name":"$H_\\beta$","type":"emission"}

    # Hgamma
    sel = df_lines["ion"] == 'Hγ' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{gamma}"]={"wls":wls,"name":"$H_\\gamma$","type":"emission"}
    
    # Hdelta
    sel = df_lines["ion"] == 'Hδ' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{delta}"]={"wls":wls,"name":"$H_\\delta$","type":"emission"}
    
    # Hepsilon
    sel = df_lines["ion"] == 'Hε' 
    wls=df_lines[sel]["wl"].values
    lines_to_plot["H{epsilon}"]={"wls":wls,"name":"$H_\\epsilon$","type":"emission"}
    
    
    sel = df_lines["ion"] == '[O II]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["[OII]"]={"wls":wls,"name":"$[OII]$","type":"emission"}
    
    
    sel = df_lines["ion"] == '[O III]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["[OIII]"]={"wls":wls,"name":"$[OIII]$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'O IV]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["[OIV]"]={"wls":wls,"name":"$[OIV]$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'O VI'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["[OVI]"]={"wls":wls,"name":"$[OVI]$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Mgb'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Mgb"]={"wls":wls,"name":"$Mgb$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Mg II]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["MgII"]={"wls":wls,"name":"$MgII$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Fe43'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Fe43"]={"wls":wls,"name":"$Fe43$","type":"emission"}
    
    sel = df_lines["ion"] == 'Fe45'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Fe45"]={"wls":wls,"name":"$Fe45$","type":"emission"}
    
    sel = df_lines["ion"] == 'Ca44'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ca44"]={"wls":wls,"name":"$Ca44$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Ca44'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ca44"]={"wls":wls,"name":"$Ca44$","type":"emission"}
    
    sel = df_lines["ion"] == 'E'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["E"]={"wls":wls,"name":"$E$","type":"emission"}
    
    sel = df_lines["ion"] =='Fe II'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["FeII24"]={"wls":wls,"name":"$FeII24$","type":"emission"}
    lines_to_plot['FeII26']={"wls":wls,"name":"$FeII26$","type":"emission"}
    
    
    lines_to_plot['weak']={"wls":[],"name":"$weak$","type":"break"}
    lines_to_plot['?']={"wls":[],"name":"$?$","type":"break"}
    
    lines_to_plot['4000{AA}-break']={"wls":[4000.],"name":"$Bal$","type":"break"}
     
    sel = df_lines["ion"] == 'Lyα'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ly{alpha}"]={"wls":wls,"name":"$Ly_\\alpha$","type":"emission"}
    
    sel = df_lines["ion"] == 'Lyβ'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ly{beta}"]={"wls":wls,"name":"$Ly_\\beta$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Lyδ'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ly{delta}"]={"wls":wls,"name":"$Ly_\\delta$","type":"emission"}
    
    
    sel = df_lines["ion"] == 'Lyε'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["Ly{epsilon}"]={"wls":wls,"name":"$Ly_\\epsilon$","type":"emission"}
    
    sel = df_lines["ion"] == 'C IV'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["CIV"]={"wls":wls,"name":"$CIV$","type":"emission"}
    
    sel = df_lines["ion"] == 'Al III'
    wls=df_lines[sel]["wl"].values
    lines_to_plot["AlIII"]={"wls":wls,"name":"$AlIII$","type":"emission"}
    
    
    sel = df_lines["ion"] == '[Ne III]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['NeIII']={"wls":wls,"name":"$NeIII$","type":"emission"}
    
    sel = df_lines["ion"] == 'He I'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['HeI']={"wls":wls,"name":"$HeI$","type":"emission"}
    
    sel = df_lines["ion"] == 'N III'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['NIII']={"wls":wls,"name":"$NIII$","type":"emission"}
    
    sel = df_lines["ion"] == 'Al II'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['AlII']={"wls":wls,"name":"$AlII$","type":"emission"}
    
    sel = df_lines["ion"] == 'Al III'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['AlIII']={"wls":wls,"name":"$AlIII$","type":"emission"}
    
    
    sel = df_lines["ion"] == '[N II]'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['NII']={"wls":wls,"name":"$NII$","type":"emission"}
    
    sel = df_lines["ion"] == 'C III'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['CIII']={"wls":wls,"name":"$CIII$","type":"emission"}
    
    sel = df_lines["ion"] == 'C IV'
    wls=df_lines[sel]["wl"].values
    lines_to_plot['CIV']={"wls":wls,"name":"$CIV$","type":"emission"}
    
    sel = df_sdss_lines["species"] == 'Si IV + O IV'
    wls=df_sdss_lines[sel]["wl"].values
    lines_to_plot['SiIV/OIV']={"wls":wls,"name":"$SiIV/OIV$","type":"emission"}
    
    lines_to_plot["(QSO)"] = {"wls":[],"name":"$QSO$","type":"emission"}
    lines_to_plot["QSO"] = {"wls":[],"name":"$QSO$","type":"emission"}
    
    lines_to_plot['NaD'] = {"wls":[],"name":"$NaD$","type":"emission"}
    
    lines_to_plot['broad'] = {"wls":[],"name":"$broad$","type":"emission"}
    
    return lines_to_plot

if __name__ == "__main__":
    # execute only if run as a script
    build_emissionlinesdict()





