# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random as rm

import random
import random as rm
import itertools
# Graficos
import pandas as pd
from IPython.display import display

import numpy as np

import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.lines as mlines

#import probscale
from decimal import Decimal
from datetime import datetime as dt

import datetime
############### FORMATO #########################
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams["mathtext.fontset"]='dejavusans'
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.sans-serif': 'Arial'})

pd.set_option("display.max_rows", 20
              , "display.max_columns",None)

######################################33



def func_tilde(x):

    di={'Limari':'Limarí',
         'Rio':'Río',
           'Fuera':'F.A.E',
           'Campana':'Campaña',
           'Vina':'Viña',
           'Caren':'Carén',
           'quebrada':'Quebrada',
           'Guatulame':'Guatulame',
           'Fuera del área de estudio':'F.A.E',
           'Subterranea':'Subterránea',
           'Criosfera':'Criósfera',
           'Precipitacion':'Precipitación'}
    
    a=[x] if x in di.keys() else x.split(' ')
    #a=x.split(' ')
    b=''
    c=0
    for i in range(len(a)):
        if i==0: a[i]=a[i][0].upper()+a[i][1:]
        if c==0: s=''
        else:s=' '
        try: 
            b+=s+di[a[i]]
            c+=1
        except: 
            b+= s+a[i]
            c+=1
    return b

def func(xx, pos=''):  # formatter function takes tick label and tick position
    if xx<0:
        x=-xx
    else: x=xx
    s = '%d' % x
    
    xsi=Decimal(str(x))%1 
        
    if xsi != Decimal(str(0)):
        
        if len(str(xsi))>5 and (Decimal(xsi)-Decimal(str(xsi)[:-1]))<Decimal('0.0000000000001'):
            xsi=float(Decimal(str(xsi)[:-1]))
            if xsi != 0: coma, dc=(',',str(xsi))
        else: coma, dc=(',',str(xsi))
    else:
        coma, dc= ('','')
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    if xx<0: return '-'+s + '.'.join(reversed(groups))+coma+dc[2:]   
    else: return s + '.'.join(reversed(groups))+coma+dc[2:]

def C_BD(DATA3):
    for k in DATA3.columns:
        if ',,' in k: DATA3.drop(columns=k, inplace=True)
    for k in DATA3.columns:
        if 'mg/l' in k:
            
            for i in list(DATA3.index):
                m=DATA3.at[i,k]
                if  type(m)==str:
                    if m == '#N/D':
                        DATA3.at[i,k]=np.nan
                        continue
                    try:
                        n=float(m)
                        DATA3.at[i,k]=n
                    except:
                        if '<' in m:
                            if ',' in m:
                                my=m.find('<')+1
                                num=m[my:].split(',')
                                n=float(num[0]+'.'+num[1])/2
                                DATA3.at[i,k]=n
                            else:
                                my=m.find('<')+1
                                num=m[my:]
                                n=float(num)/2
                                DATA3.at[i,k]=n
                        elif ',' in m:
                            my=m.find('<')+1
                            num=m[my:].split(',')
                            n=float(num[0]+'.'+num[1])/2
                            DATA3.at[i,k]=n
                        else:
                            my=m.find('<')+1
                            num=m[my:]
                            n=float(num)/2
                            DATA3.at[i,k]=n
            cop=DATA3[k].copy()
            DATA3[k]=cop.astype('float')
            if 'arbonat' in k or 'CO3' in k:
                DATA3[k].fillna(value=0,inplace=True)
    return DATA3

def ncolorandom2(j,cc='tab20b'):
    colores=list()
    clp=cm.get_cmap(cc) if j <=26 else cm.get_cmap('gist_rainbow')
    for i in range (0,j):
        if j==1: colores.append([[1,0,0,1]])
        else: colores.append([list(clp(i/(j-1))),])
    #n=len(colores)
    """ret=list()
    for i in range(0,j,1):
        ret.append(next(colores))"""
    #print (colores)
    return colores
def numb_sp(df,n):
    l=len(df)
    n1=float(n)
    num=int(str(l/n1+1).split('.')[0])
    x=[num,int(n)]
    lista=list()
    for i in range(1,l+1):
        x.append(i)
        lista.append(tuple(x))
        x=[num,int(n)]
    return lista

def norma(df=pd.DataFrame(),s=12,fecha='Fecha',ultimo='',primero=''):
    # Example data
    if not fecha in df.columns: 
        #print ('if 161')
        return False
    #print('norma 163')
    if ultimo == '': MF= max(df[fecha])
    else: MF= ultimo
    if primero == '': NF= min(df[fecha])
    else: NF= primero
    #NF=min(df[fecha])
    lst=[NF,MF]
    dff=pd.DataFrame()
    for f in lst:
        data1 = {'Cod_Muestra' : ['Nch 409', 'Nch 1333'],
                'Tipo'  : ['Nch 409', 'Nch 1333'],
                fecha:[f,f],
                'Al'     : [0.2, np.nan],
                'Cu'     : [2.0, 0.2],
                'Cr'     : [0.05, 0.1],
                'F'     : [1.5, 1.0],
                'Fe'     : [0.3, 5.0],
                'Mn'      : [0.1, 0.2],
                'Mg'   : [125.0, np.nan],
                'Se'    : [0.01, 0.02],
                'Zn'    : [3.0, 2.0],
                'As'    : [0.01, 0.1],
                'Cd'    : [0.01, 0.01],
                'Hg'    : [0.001, 0.001],
                'NO3'    : [50, np.nan],
                'Pb'    : [0.05, 5],
                'Cl'    : [400, 200],
                'SO4'    : [500, 250],
                'TDS'    : [1500, 5000],
                'pH'     : [6.5, 8.5]
                }
        dff=pd.concat([pd.DataFrame(data1),dff.copy()])
    #df2 = pd.concat([dff,df])
    
    dff.reset_index(inplace=True, drop=True)
    #print (df3['Mg'])
    # df = pd.read_csv('../data/data_template.csv')
    return dff
def plot_promedio(ax,prom,xmin,xmax,q1='',q2='',col='yellow',aph=0.3,qts=True):
    x=np.array([xmin,xmax])
    y=np.array([prom,prom])
    yq1=np.array([q1,q1])
    yq2=np.array([q2,q2])
    ax.plot(x,y,color='grey',ls='-',zorder=1)
    if qts:
        ax.plot(x,yq1,color='grey',ls=':',zorder=0)
        ax.plot(x,yq2,color='grey',ls=':',zorder=0)
        ax.fill_between(x,yq1,yq2,alpha=aph, color=col)

def grafico_2v(Bal=pd.DataFrame(),
                v1="rCa",v2="",xv=['pH','TDS','As','SO4','Cu','Fe','Mn','Se','Cl','Pb','Al'],
                filt=['Tipo'],fecha_filt=(True,"01-08-2022"),
                LOGY=True, LOGX=False,
                ss=12,xlim="",ylim="",
                f=False,inv=False,AXS=(20,15),
                leer=False, save_dic=False,
                zoom=(False,{'ax':'',
                        'xlim':'',
                        'ylim':'',
                        'z_dom':'',
                        'z_ubi':'',
                        'z_dim':''})):

    
    xlist=isinstance(xv,list)

    y_format = mticker.FuncFormatter(func)
    f_format=mticker.FuncFormatter(func_fecha)
    #print (y_format)

    dic_titulos=diccionario_titulos(xlist,v1,v2,xv)

    if not xlist:
        absi=xv
        esfecha=isinstance(Bal.at[Bal.index[0],xv],datetime.date)
        xv=[xv]
        
    else: esfecha= False
    
    listael=xv
    listaelmen=listael#['SO4','Fe','Se','Cu']
    contador=0
    Line_width=1
    if isinstance(AXS,tuple):
        w,h=AXS
        sf=plt.figure(figsize=(w/2.54,h/2.54),dpi=150)
        ind=0
        if xlist: sf.set_figheight(h*len(listael))
        else: sf.set_figheight(h)
        Line_width=w/15*0.8
    num=numb_sp(listael,1)
    
    if fecha_filt[0]:
        try:
            ff= dt.strptime(fecha_filt[1], "%d-%m-%Y")
            P=Bal.loc[(Bal['Fecha'] >= ff)].reset_index(drop=True)
            Nch=norma(P)
        except: 
            pass
        
    else:
        #print ('else 263') 
        P=Bal.copy()
        Nch=norma(P)
    
    if esfecha:
        try:
            P.sort_values(by='Fecha',inplace=True)
            x_min= dt.strptime(fecha_filt[1], "%d-%m-%Y")
            x_max= datetime.datetime(2022,10,1)
            Nch=norma(P,ultimo=x_max,primero=x_min)
            print (f'Gráfico {v1}_Fecha')
        except: pass
    else: 
        #print (esfecha)
        estadis=Bal.describe()[listaelmen].T#.at['SO4','mean']
    # Eliminar val Nan
    if v2!="": drp=[v1,v2]
    else: drp=[v1]
    P.dropna(subset=xv+drp, axis=0, inplace=True)
    if isinstance(filt,list):
        try:
            dfg=P.groupby(filt)
            kg=dfg.groups.keys()
            if len(filt)>1:
                cla= {'Clase1': filt[0],
                    'Clase2': filt[1] }
                
            else: 
                cla= {'Clase1': filt[0],
                'Clase2': filt[0] }
            #dic_tmp, dic_mar = ncolran_dic(P,cla,T=(len(filt)>1))#dict(zip(list(kg),ncolorandom(len(list(kg)))))
            
            clasi=True
        except: return print (f'Error: No se puede clasificar por {filt}')
        dic_tmp, dic_mar = ncolran_dic(P,cla,T=(len(filt)>1), leer=leer, save=save_dic)

            

        lgdM=list()
        lgdC=list()
        
        fc=1.5
        if cla['Clase1']==cla['Clase2']:
            for dc in dic_mar.keys():
                if isinstance(dc,datetime.date): ndc= conv_fecha(dc)
                elif dc == '*':
                    ndc=dc
                    sss=ss*fc**2
                else:
                    sss=ss
                    ndc=dc
                if 'et al.,' in ndc:
                    ndc=ndc.replace('et al.,','$\it{et}$'+' '+'$\it{al.,}$')
                ndc=func_tilde(ndc)
                lgdM.append(mlines.Line2D([],[],markerfacecolor=dic_tmp[dc][0],markeredgecolor='k',marker=dic_mar[dc],
                                          linestyle='None', markersize=sss, label=ndc))
        else:
            for dc in dic_mar.keys():
                if isinstance(dc,datetime.date): ndc= conv_fecha(dc)
                elif dic_mar[dc] == '*':
                    ndc=dc
                    sss=ss*fc
                    #print (ss,sss)
                else:
                    sss=ss
                    ndc=dc
                if 'et al.' in ndc:
                    ndc=ndc.replace('et al.,','$\it{et}$'+' '+'$\it{al.,}$')
                ndc=func_tilde(ndc)
                lgdM.append(mlines.Line2D([],[],markerfacecolor='none',markeredgecolor='k',marker=dic_mar[dc],
                                          linestyle='None', markersize=sss, label=ndc))
            for dc in dic_tmp.keys():
                if isinstance(dc,datetime.date): ndc= conv_fecha(dc)
                
                else:
                    ndc=dc
                if 'et al.' in ndc:
                    ndc=ndc.replace('et al.,','$\it{et}$'+' '+'$\it{al.,}$')
                    #print(ndc)
                lgdC.append(mlines.Line2D([],[],color=dic_tmp[dc][0],marker='s',markeredgecolor='k',linestyle='None', markersize=sss, label=ndc))
        
        lgd_c=lgdM+lgdC
    else: 
        kg=0
        clasi=False
    
    
    for elemento in listael:
        
        

        if not isinstance(AXS, tuple) and AXS!="":
            ax=AXS
            
        else: ax=sf.add_subplot(num[contador][0],num[contador][1],num[contador][2])
        contador+=1
        if v2 =="" and v1!="":
            div=False
        elif v1=="" and v2 =="":
            return ("Valor y inválido")
        else: div=True

        absi=elemento
        gim=0
        ## ZOOM FUNC ####
        if zoom[0]:
            DF=zoom[1]
            DF['ax']=ax
            DCG= {'Bal':Bal,
                'v1':v1,
                'v2':v2,
                'xv':xv,
                'filt':filt,
                'LOGY':LOGY,
                'LOGX':LOGX,
                'f':False,
                'ss':ss, 
                'inv':inv,
                'xlim':xlim}

            ax=graf_zoom(DF,logx=LOGX,logy=LOGY,dic_gp=DCG,leer=leer)
        #FIN ZOOM FUNC
        for i in list(kg):
            if i ==0: df=P
            else: df=dfg.get_group(i)
            #print (i,df[ord])
            if v2!="": drp=[v1,v2]
            else: drp=[v1]
            df.dropna(subset=xv+drp, axis=0, inplace=True)
            #print (df)
            if len(df.index)==0: continue #or df.at[df.index[0],'Tipo']=='CANDELARIA': continue
            #if df.at[df.index[0],'Tipo']=='CANDELARIA': print (df[ord])
            nombre=""
            esp=""
            for i in filt:
                if isinstance(df.at[df.index[0],i],datetime.date):
                    ad=conv_fecha(df.at[df.index[0],i])
                    
                else: ad = df.at[df.index[0],i]
                nombre+=esp+ad
                esp=" - "
            if clasi:
                #print (dic_mar)
                color=dic_tmp[df.at[df.index[0],cla['Clase1']]]
                mar=dic_mar[df.at[df.index[0],cla['Clase2']]]
            else: color, mar=('r','o')
            x=df[absi]
            ord= not div
            if div: 
                Y=df[v2]/df[v1]
                
            else:
                Y=df[v1]
            if mar == '*':
                sst=ss*fc
            else:
                sst=ss   
            
            alp=1
            if i == 'Pozo':
                zord=0
                color='gray'
            else: zord = 2
            if esfecha:
                color=color[0]
                ax.plot(x,Y,label=nombre,ls='--',ms=sst,mfc=color,marker=mar,mec='k', alpha=alp, zorder=zord)
                
            elif inv: ax.scatter(Y,x,label=nombre,s=sst**2,c=color,marker=mar,edgecolors='k',linewidth=Line_width, alpha=alp, zorder=zord)
            else: ax.scatter(x,Y,label=nombre,s=sst**2,c=color,marker=mar,edgecolors='k',linewidth=Line_width, alpha=alp, zorder=zord)
            for i in listaelmen:
                    if ord and i == v1 and esfecha:
                        
                        plot_promedio(ax=ax,prom=estadis.at[i,'mean'],xmin=x_min,xmax=x_max,q1=estadis.at[i,'75%'],
                        q2=estadis.at[i,'25%'],col='gray',aph=0.05, qts=False)
                        if gim==0:
                            if v1 in ['TDS', 'Cl', 'pH']: 
                                ubic= 'bottom'
                                ct=0.01*estadis.at[i,'mean']
                            else: 
                                ubic='top'
                                ct=-0.1*estadis.at[i,'mean']
                            plt.annotate('Promedio cuenca río Copiapó',(x_max,estadis.at[i,'mean']+ct),
                            fontsize=13,c='gray',rotation=0,ha='right',va=ubic)
                            gim=1
                        #print (ord ,estadis.at[i,'mean'],estadis.at[i,'75%'],estadis.at[i,'25%'])
            
        if not isinstance(AXS, tuple): return
        sf.tight_layout() 
        if not isinstance(Nch, bool):
            ng=Nch.groupby('Tipo')
            for ngi in list(ng.groups.keys()):
                Xg=ng.get_group(ngi)
                if not ord: continue
                if '409' in ngi: col='black'
                else: col= 'red'
                try:
                    xn=Xg[x]
                    yn=Xg[v1]
                    if v1== 'Al': etiq='OMS'
                    else: etiq=ngi
                    plt.annotate(etiq,(x_max,Xg.at[Xg.index[0],ord]),fontsize=13,c=col ,rotation=0,ha='right',va='bottom')
                    plt.plot(xn,yn,c=col,ls='--',zorder=1)
                    
                    #print(min(xn))
                except: continue
        if not inv:
            X=absi
            if ord:
                Y=v1
            else:
                P[v2+'/'+v1]=P[v2]/P[v1]
                Y=v2+'/'+v1
        else: 
            Y=absi
            if ord:
                X=v1
            else:
                P[v2+'/'+v1]=P[v2]/P[v1]
                X=v2+'/'+v1
#Definir limites de los 3 graficos

        if isinstance(xlim,tuple):# and xlim[0]!=xlim[1]:
            x_min=xlim[0]
            x_max=xlim[1]
        else:
            x_mini=min(P[X].dropna())
            x_maxi=max(P[X].dropna())
            espx=(x_maxi-x_mini)*0.1
            if LOGX:
                x_min,x_max= x_mini*0.5, x_maxi*5
            else:
                x_min=x_mini-espx
                x_max=x_maxi+espx
        if isinstance(ylim,tuple):# and ylim[0]!=ylim[1]:
            y_min=ylim[0]
            y_max=ylim[1]
        else:
            y_mini=min(P[Y].dropna())
            y_maxi=max(P[Y].dropna())
            espy=(y_maxi-y_mini)*0.1
            if LOGY:
                y_min,y_max= y_mini*0.5, y_maxi*5
            else:
                y_min=y_mini-espy
                y_max=y_maxi+espy

        if LOGY and ord!='pH':
            ax.semilogy()
            
        if LOGX:
            ax.semilogx()
            
        ax.set_ylim(y_min,y_max)
        ax.set_xlim(x_min,x_max)
        
        
            
        
        
        
        


        ## unidades f
        #if absi[0]== 'r':unix=" [meq/l]"
        #elif absi=='Fecha':unix=''
        #else: unix=" [mg/l]"
        #if v1[0]== 'r':uniy=" [meq/l]"
        #elif v1[0]== '%':  uniy=""
        #elif v1[0:2]== 'IS':  uniy=""
        #else: uniy=" [mg/l]"
        unix,uniy='',''
        if not inv:

            if absi=='pH': ax.set_xlabel(dic_titulos[absi])
            elif ord:
                ax.set_ylabel(dic_titulos[v1] +uniy)
                ax.set_xlabel(dic_titulos[absi]+unix)
            else: 
                ax.set_ylabel(dic_titulos[v2]+"/"+dic_titulos[v1])
                ax.set_xlabel(dic_titulos[absi]+unix)
        else:
            if absi=='pH': ax.set_ylabel(dic_titulos[absi])
            elif ord:
                
                ax.set_ylabel(dic_titulos[absi]+unix)
            else: 
                
                ax.set_ylabel(dic_titulos[absi]+unix)
            if contador == len(listael) :
                
                if ord:
                    ax.set_xlabel(dic_titulos[v1] +uniy)
                    
                else: 
                    ax.set_xlabel(dic_titulos[v2]+"/"+dic_titulos[v1])
                    
            
        if esfecha: 
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
            plt.xticks(rotation=45)
        else:
            ax.xaxis.set_major_formatter(y_format)

        ax.get_yaxis().set_major_formatter(y_format)
        
        
        if contador == len(listael) and not f:
            lgd1=plt.legend(handles=lgd_c,bbox_to_anchor=(0.5, -0.1),borderaxespad=1,loc='upper center', markerscale=1, frameon=True,
                    labelspacing=0.25,ncol=3)
            ax.add_artist(lgd1)

        

        


        #print (mm,(maxi))

        #cwd = os.getcwd()
    if not xlist and not div and not f:
        
        plt.savefig(f'scatter_{v1}_{xv[0]}.jpg',bbox_extra_artists=(lgd1,),bbox_inches='tight')
    elif not xlist and not f:
        plt.savefig(f'scatter_{v1}_lista.jpg',bbox_extra_artists=(lgd1,),bbox_inches='tight')
    elif xlist and not f:
        plt.savefig(f'scatter_{v1}_{v2}_lista.jpg',bbox_extra_artists=(lgd1,),bbox_inches='tight')
    else: a=1

    if not f: return 'listo'
    else: return ax, sf, [lgdM,lgdC]

dic_marb={'IM270 G4': 'd',
         'IM270 G12': 'h',
         'IM200': '^',
         'IM60' : 'D',
         'Pozo': 'o',
         'Río' : 's',
         'Pozo 8': 'P',
         'P12':'p',
         'HA01':'p',
         'P14':'p',
         'Vertiente': 'v',
         'Noria':'o',
         'IM135' : 'd',
         'Pozo Ag. Copayapu' : 'h',
         'Pique': '*',
         'IM256':'v',
         'IM300':'*',
         'IM170':'D',
         'IM80':'D',
         'CANDELARIA': 'P'}
         
def diccionario_titulos(xlist,v1,v2,xv):

    lista1=['Balan_Ionico','Oxigeno_18','Cl_mg_l',' CE_terr ',' Deuterio ','TSD',' Oxigeno_18 ','IS','Fecha','Deuterio','O18','Na','Balance %','HCO3','pH','Al','TDS','Ca','Cu', 'Cr','F', 'Fe', 'Mn', 'Mg', 'Se', 'Zn','As','Cd','Hg','NO3','Pb','Cl','SO4','TDS','Ferrico']
    lista2=['Balance %',"$\delta^{18}$ O‰ V-SMOW","Cloruro [mg/l]","Conductividad eléctrica [$\mu$S/cm]","$\delta^{2}$ H‰ V-SMOW",'Sólidos totales disueltos [mg/l]',"$\delta^{18}$ O‰ V-SMOW",'IS yeso','Fecha',"$\delta^{2}$ H‰ V-SMOW","$\delta^{18}$ O‰ V-SMOW",'Na$^{+}$','Balance [%]','HCO3$^{-}$','pH','Al','Sólidos totales disueltos ','Ca$^{+2}$','Cu', 'Cr', 'F$^-$', 'Fe$_{tot}$', 'Mn','Mg$^{+2}$', 'Se', 'Zn','As','Cd','Hg','NO3$^{-2}$','Pb','Cl$^-$','SO$_4^{-2}$','TDS','Fe$^{+3}$']
    dic_titulos=dict(zip(lista1,lista2))
    if v1 not in lista1:
        if v1[1:] in lista1: dic_titulos[v1]=dic_titulos[v1[1:]]
        else:
            #print (v1[1:])
            dic_titulos[v1]= input(f"Ingrese Etiqueta para el valor {v1}")

    
    if v2 !="" and v2 not in lista1:
        if v2[1:] in lista1: dic_titulos[v2]=dic_titulos[v2[1:]]
        else: dic_titulos[v2]= input(f"Ingrese Etiqueta para el valor {v2}")

    if not xlist and xv not in lista1:
        if xv[1:] in lista1: dic_titulos[xv]=dic_titulos[xv[1:]]
        else: dic_titulos[xv]= input(f"Ingrese Etiqueta para el valor {xv}")

    elif xlist:
        for x_lab in xv:
            if  x_lab not in lista1:
                if x_lab[1:] in lista1: dic_titulos[x_lab]=dic_titulos[x_lab[1:]]
                else: dic_titulos[x_lab]=input(f"Ingrese Etiqueta para el valor {x_lab}")

    return dic_titulos

def func_fecha(ff,pos):
    a=conv_fecha(ff)
    #print (a)
    return  a


def conv_fecha(i):
    mm=str(i)[5:7]
    dd=str(i)[8:10]
    aa=str(i)[2:4]
    #print (i,mm+dd+aa)
    return (dd+'-'+mm+'-'+aa)

def grafico_bal(Bal=pd.DataFrame(), v1='Balance %',xv=['TDS'], filt=['Tipo'],fecha_filt=(True,"01-08-2022"),LOGY=True, LOGX=True, ss=12, leer=False,save_dic=False,tam=(5,5)):
    
    y_format = mticker.FuncFormatter(func)
    plt.rcParams['image.cmap'] = "bwr"
    plt.rcParams['figure.dpi'] = "100"
    plt.rcParams['savefig.bbox'] = "tight"
    #style.use('default') or plt.style.use('default')
    #plt.rcParams.update({'font.size': 12})
    #plt.rcParams.update({'font.sans-serif': 'Arial'})
    if LOGY:
        Bal[v1]=abs(Bal[v1].copy())
    axs, sf,lgd_c = grafico_2v(Bal=Bal, v1=v1,xv=xv,filt=filt,fecha_filt=fecha_filt,LOGY=LOGY, LOGX=LOGX, ss=ss,f=True,leer=leer,save_dic=save_dic)
    axs.plot([0,80000],[10,10],c='black',ls=':')
    #Lineas
    sf.set_figwidth(tam[0])
    sf.set_figheight(tam[1])

    lim=[100,500,1000,5000,20000,50000]
    lim2=[0,100,500,1000,5000,20000,50000,80000]
    alt=50
    tit=[('Dulce',alt),(' Dulce\n moderadamente\n mineralizada',alt),
        ('Dulce\n mineralizada',alt),('Salobre',alt),('Salada',alt),('Muy salada',alt),('Salmuera',alt)]
    for i in lim:
        auy=np.asarray([i,i])
        aux=np.asarray([-0.5,50])
        plt.plot(auy,aux,c='black',ls='--')
    for i,y in tit:
        if len(i)<8:
            rota=0
        else: 
            rota=90
        c=tit.index((i,y))
        plt.annotate(i,((lim2[c]+lim2[c+1])*0.66,y), rotation=rota,ha='right',va='top')
    plt.annotate('10%',(2.2,11), rotation=0,ha='right',va='bottom')
    axs.set_ylim(0.024,55)
    axs.set_xlim(0.2,90000)
    axs.semilogx()
    axs.semilogy()
    axs.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axs.yaxis.set_major_formatter(mticker.ScalarFormatter())
    axs.get_xaxis().set_major_formatter(y_format)
    axs.get_yaxis().set_major_formatter(y_format)
    axs.set_ylabel("Error de balance absoluto [%]")
    axs.set_xlabel("Sólidos totales disueltos [mg/l]")
    #axs.legend(ncol=2,fontsize=13, loc=1)
    axs.grid()
    #lgd1=plt.legend(handles=lgd_c,bbox_to_anchor=(0.5, -0.2),borderaxespad=1,loc='upper center', markerscale=1, frameon=True, fontsize=16,
    #                labelspacing=0.25,ncol=4)
    #axs.add_artist(lgd1)
    #print (lgd_c[0], lgd_c[1])
    #lgd1=plt.legend(lgd_c[0], lgd_c[1],bbox_to_anchor=(0.5, -0.15),borderaxespad=1,loc='upper center', markerscale=1, frameon=True, fontsize=16,
    #                labelspacing=0.25,ncol=3)
    #lgd2=plt.legend(lgd_c[2], lgd_c[3],bbox_to_anchor=(0.5, -0.15),borderaxespad=1,loc='upper center', markerscale=1, frameon=True, fontsize=16,
    #                labelspacing=0.25,ncol=3)
    #axs.add_artist(lgd1)
    #axs.add_artist(lgd2)

    sf.savefig('scatter_{}.jpg'.format('Balan_abso vs TSD_tipob' ),bbox_inches='tight')
    return lgd_c


def ncolran_dic(Y_df,cla,T=True, leer=False,save=False):

    
    
    simb=list(Line2D.filled_markers)
    remv=[',','None',' ','8']
    for r in remv:
        try:simb.remove(r)
        except: continue
    col=list(mcol.cnames.items())
    clases1=list(Y_df.groupby([cla["Clase1"]]).groups.keys())
    clases2=list(Y_df.groupby([cla["Clase2"]]).groups.keys())
    colores=ncolorandom2(len(clases1))
    
    simbolos=ncolorandom_ord(len(clases2),simb)
    colores_2=list()
    simbolos_2=list()
    #Guardar Cargar
    if leer:
        try:
            df1=pd.read_csv('Dict_col.csv')
            v=[[[r,g,b,1]]for (r,g,b) in zip(df1['R'],df1['G'],df1['B'])]
            k=list(df1[cla["Clase1"]])
            dcl1=dict(zip(k,v))
            colores_2=[dcl1[x] for x in clases1 if x in list(df1[cla["Clase1"]])] 
            clases1_2=[x for x in clases1 if x in list(df1[cla["Clase1"]])]
            
        except: 
            print ('Error al cargar dic de colores, se crea uno nuevo')
            colores_2=colores
            clases1_2=clases1
        try:
            df2=pd.read_csv('Dict_sim.csv')
            v=df2['simbolos']
            k=list(df2[cla["Clase2"]])
            dcl2=dict(zip(k,v))
            simbolos_2=[dcl2[x] for x in clases2 if x in list(df2[cla["Clase2"]])]
            clases2_2=[x for x in clases2 if x in list(df2[cla["Clase2"]])]
            
        except: 
            print ('Error al cargar dic de simbolos, se crea uno nuevo')
            simbolos_2=simbolos
            clases2_2=clases2
    if leer and len(clases1)> len(colores_2):
        lista=[x for x in clases1 if x not in clases1_2]
        print (f'no existe definicion de color para los siguientes elementos {lista}')
    elif leer and len(clases1)<= len(colores_2): colores, clases1= colores_2, clases1_2
    if leer and len(clases2)> len(simbolos_2):
        lista=[x for x in clases2 if x not in clases2_2]
        print (f'no existe definicion de simbolos para los siguientes elementos {lista}')
    elif leer and len(clases2) <= len(simbolos_2): simbolos, clases2= simbolos_2, clases2_2
    if save:          
        df1=pd.DataFrame({cla["Clase1"]:clases1, 'R': [j[0][0] for j in colores], 'G':[j[0][1] for j in colores], 'B':[j[0][2] for j in colores], 'A':[j[0][3] for j in colores]} )
        df2=pd.DataFrame( {cla["Clase2"]:clases2, 'simbolos': simbolos} )
        df1.to_csv('Dict_col.csv')
        df2.to_csv('Dict_sim.csv')
    

    
    dict_col=dict(zip(clases1,colores))
    if T: dict_sim=dict(zip(clases2,simbolos))
    else:
        dict_sim=dict(zip(clases2,['o']*len(clases2)))
    
    return dict_col, dict_sim

def ncolorandom(j,lista=list(mcol.cnames.items())):
    val=isinstance(lista[0],tuple)
    n=len(lista)
    ret=list()
    if val:
        for i in rm.sample(range(0,n),k=j):
            ret.append(lista[i][0])
    else:
        for i in rm.sample(range(0,n),k=j):
            ret.append(lista[i])
    return ret

def ncolorandom_ord(j,lista=list(mcol.cnames.items())):
    val=isinstance(lista[0],tuple)
    n=len(lista)
    if len(lista)<j:
        listafin=lista*(int(j/len(lista))+1)
    else: listafin=lista
    ret=list()
    if val:
        for i in rm.sample(range(0,n),k=j):
            ret.append(lista[i][0])
    else:
        ret=listafin[0:(j)]
       
    return ret

    ########### Funcion de ISOTOPOS #########################
#########################################################

def graf_isotopos(ax,rectas_aux=dict(),xlim=(-15,0),ylim=(-110,0),zoom=True,z_dom=(-11,-9.3,-88,-73),z_ubi=(-14,-40),z_dim=(6,40),dic_gp=dict(),leer=False):
    if rectas_aux == dict():
        rectas_aux={'Línea de evaporación': (5.454,-23.89,':','\n(Troncoso $\it{et al.,}$ 2012)','k'),
                    'Línea Meteórica Local': (8.07,+13.5,'--','\n(Troncoso $\it{et al.,}$ 2012)','k'),
                    'Línea Meteórica Mundial': (8,10,'-','\n(Craig, 1961)','k')}
        return rectas_aux
    if dic_gp == dict():
        return ('Error, Ingrese variables del grafico en forma de diccionario')
    axl=[ax]
    auxc=np.asarray([0.01,1000])
    a=rectas_aux['Línea de evaporación'][0]
    b=rectas_aux['Línea de evaporación'][1]
    auxc[0]=-(rectas_aux['Línea Meteórica Local'][1]-b)/(+rectas_aux['Línea Meteórica Local'][0]-a)
    dxlim=xlim[1]-xlim[0]
    #print (aux[0])
    #xl=aux[0]
    xl=xlim[0]
    xf=xl+dxlim

    

    if zoom:
        yl, yf = xl*rectas_aux['Línea Meteórica Local'][0]+rectas_aux['Línea Meteórica Local'][1], xf*rectas_aux['Línea Meteórica Local'][0]+rectas_aux['Línea Meteórica Local'][1]
        

        x1,x2,y1,y2 =z_dom[0],z_dom[1],z_dom[2],z_dom[3]


        axin_x, axin_y = z_ubi
        largo, alto = z_dim
        
        axins=ax.inset_axes([1-(xf-(axin_x))/(xf-xl),1-(yf-(axin_y))/(yf-yl),largo/(xf-xl),alto/(yf-yl)]) #[1-abs(axin_x)/(xf-xl),1-abs(axin_y)/(yf-yl),0.47,0.47]
        
        
        
        axl.append(axins)
        ax.set_xlim(xl,xf)
        #ax.set_ylim(yl,yf)
        ax.set_ylim(ylim[0],ylim[1])
        
    auxt=np.asarray([-20,12.5])
    aux=np.asarray([-20,12.5])
    lgd=list()
    for ky in rectas_aux.keys():
        lgd.append(mlines.Line2D([],[],color=rectas_aux[ky][4],linestyle=rectas_aux[ky][2], markersize=10, label=rectas_aux[ky][3]))
        aux[0]=auxc[0] if ky=='Línea de evaporación' else auxt[0]
        for axs in axl:
            axs.plot(aux[0:2],aux[0:2]*rectas_aux[ky][0]+rectas_aux[ky][1],c=rectas_aux[ky][4],ls=rectas_aux[ky][2],label=rectas_aux[ky][3],zorder=0)
            
    if zoom:
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xlabel("$\delta^{18}$ O‰ V-SMOW")
        axins.set_ylabel("$\delta^{2}$ H‰ V-SMOW")
        #axins.set_xticklabels([])
        #axins.set_yticklabels([])
        ax.indicate_inset_zoom(axins, edgecolor="black",lw=2)

    DCG= dic_gp
    DCGK=list(DCG.keys())
    if zoom: 
        grafico_2v(Bal=DCG[DCGK[0]],v1=DCG[DCGK[1]],v2=DCG[DCGK[2]],xv=DCG[DCGK[3]],filt=DCG[DCGK[4]],LOGY=DCG[DCGK[5]],LOGX=DCG[DCGK[6]],f=DCG[DCGK[7]],ss=DCG[DCGK[8]], inv=DCG[DCGK[9]],xlim=DCG[DCGK[10]],AXS=axins,fecha_filt=(False,""),leer=leer)
        return axins , lgd
    else: 
        #grafico_2v(Bal=DCG[DCGK[0]],v1=DCG[DCGK[1]],v2=DCG[DCGK[2]],xv=DCG[DCGK[3]],filt=DCG[DCGK[4]],LOGY=DCG[DCGK[5]],LOGX=DCG[DCGK[6]],f=DCG[DCGK[7]],ss=DCG[DCGK[8]], inv=DCG[DCGK[9]],xlim=DCG[DCGK[10]],fecha_filt=(False,""),leer=leer)
        return "",lgd
################################# FIN FUNC ISOTOPOS ##################################################
################################## FUN ZOOM #########################################################

def graf_zoom(DFK,logy=False,logx=False,dic_gp=dict(),leer=False):
    
    ax=DFK['ax']
    xlim=DFK['xlim']
    ylim=DFK['ylim']
    z_dom=DFK['z_dom']
    z_ubi=DFK['z_ubi']
    z_dim=DFK['z_dim']

    dxlim=xlim[1]-xlim[0]
    xl=xlim[0]
    xf=xl+dxlim
    yl, yf = ylim
    x1,x2,y1,y2 =z_dom[0],z_dom[1],z_dom[2],z_dom[3]
    axin_x, axin_y = z_ubi
    largo, alto = z_dim
    if logy:
        yl=np.log10(yl)
        yf=np.log10(yf)
        axin_y=np.log10(axin_y)
        alto=np.log10(alto)
    if logx: 
        xl=np.log10(xl)
        xf=np.log10(xf)
        axin_x=np.log10(axin_x)
        largo=np.log10(largo)
    axins=ax.inset_axes([1-(xf-(axin_x))/(xf-xl),1-(yf-(axin_y))/(yf-yl),largo/(xf-xl),alto/(yf-yl)]) #[1-abs(axin_x)/(xf-xl),1-abs(axin_y)/(yf-yl),0.47,0.47]
    ax.set_xlim(xl,xf)
    ax.set_ylim(yl,yf)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    
    y_format = mticker.FuncFormatter(func)
    
    if logy:
        axins.semilogy()
        #axins.minorticks_off()
        axins.yaxis.grid(True,which='both')
        
        axins.yaxis.set_minor_formatter(mticker.NullFormatter())
        axins.yaxis.set_major_formatter(mticker.ScalarFormatter())
        axins.get_yaxis().set_major_formatter(y_format)
    else: 
        axins.yaxis.grid(True,which='major')
        axins.yaxis.set_major_formatter(mticker.ScalarFormatter())
        axins.get_yaxis().set_major_formatter(y_format)

    if logx:
        axins.semilogx()
        #axins.minorticks_off()
        axins.xaxis.grid(True,which='both')
        axins.xaxis.set_minor_formatter(mticker.NullFormatter())
        axins.xaxis.set_major_formatter(mticker.ScalarFormatter())
        axins.get_xaxis().set_major_formatter(y_format)
    else:
        axins.xaxis.grid(True,which='major')
        axins.xaxis.set_major_formatter(mticker.ScalarFormatter())
        axins.get_xaxis().set_major_formatter(y_format)



    
    
    
    
    
    
    
    # 
    # 

                                          
    ax.indicate_inset_zoom(axins, edgecolor="black",lw=1)
    DCG= dic_gp
    DCGK=list(DCG.keys())
    grafico_2v(Bal=DCG[DCGK[0]],
            v1=DCG[DCGK[1]],
            v2=DCG[DCGK[2]],
            xv=DCG[DCGK[3]],
            filt=DCG[DCGK[4]],
            LOGY=DCG[DCGK[5]],
            LOGX=DCG[DCGK[6]],
            f=True,
            ss=DCG[DCGK[8]], 
            inv=DCG[DCGK[9]],
            xlim=DCG[DCGK[10]],
            AXS=axins,
            fecha_filt=(False,""),
            leer=leer)
    # sf.set_figwidth(15/2.54)
    # sf.set_figheight(10/2.54) #figsize=(w/2.54,h/2.54)
    return ax

#######################################################################################################

def df_fecha(df,col):
    samples=df.copy()
    for i in df.index:
        df.at[i,col]= dt.strptime(samples.at[i,col], "%Y-%m-%d %H:%M:%S")
    return df

def df_vis_tabla(df=pd.DataFrame(),nombre='Salida_vis'):
    for c in df.columns.values:
        
        if df[c].dtypes == 'int64' or df[c].dtypes == 'float64':
            for ind in df.index.values:
                df[c]=df[c].astype('object')
                df.at[ind,c]= func(df.at[ind,c])
    display(df)
    df.to_excel(f'{nombre}.xlsx',sheet_name='DataFrame')
    return
