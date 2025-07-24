import pandas as pd
import re
import regex
import demoji
import numpy as np
from collections import Counter
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import streamlit as st


with st.container():
    st.title('¡Feliz primer mes juntos! ❤️')
    st.subheader('Análisis especial de nuestro chat de WhatsApp')
    st.write('Cada mensaje, emoji y palabra cuenta nuestra historia. ¡Te amo Paola! 🥰')
    st.write('Un mes de novios es solo el inicio de algo con mucho futuro. 🥰')

# Carrusel de fotos de la pareja
try:
    from streamlit_carousel import carousel
    images = [
        {
            "img": f"Resources/{i}.jpeg",
            "caption": f"Foto {i}",
            "title": f"",
            "text": f""
        } for i in range(1, 9)
    ]
    st.markdown('---')
    st.subheader('📸 Nuestros momentos juntos')
    carousel(images)
except ImportError:
    st.warning('Para ver el carrusel de fotos instala streamlit-carousel: pip install streamlit-carousel')

st.markdown("<hr style='border:1px solid #d72660;'>", unsafe_allow_html=True)

# Paso 1: Definir funciones necesarias
def IniciaConFechaYHora(s):
    patron = r'^\d{1,2}/\d{1,2}/\d{2,4}\s\d{1,2}:\d{2}.*\s-\s'
    return re.match(patron, s) is not None

def ObtenerPartes(linea):
    partes = linea.split(' - ', 1)
    fecha_hora = partes[0].strip()
    resto = partes[1] if len(partes) > 1 else ''
    fecha_hora = fecha_hora.replace('\u202f', ' ').replace(' ', ' ')
    if ': ' in resto:
        Miembro, Mensaje = resto.split(': ', 1)
    else:
        Miembro, Mensaje = None, resto
    return fecha_hora, Miembro, Mensaje

# Paso 2: Obtener el dataframe usando el archivo txt y las funciones definidas
RutaChat = 'Data/Chat de WhatsApp con Paola Rousse ❤️.txt'
with open(RutaChat, encoding="utf-8") as fp:
    lineas = fp.readlines()
DatosLista = []
FechaHora, Miembro, Mensaje = None, None, ""
for linea in lineas:
    linea = linea.strip()
    if IniciaConFechaYHora(linea):
        if Mensaje and Miembro:
            DatosLista.append([FechaHora, Miembro, Mensaje])
        FechaHora, Miembro, Mensaje = ObtenerPartes(linea)
    else:
        Mensaje += " " + linea
if Mensaje and Miembro:
    DatosLista.append([FechaHora, Miembro, Mensaje])
df = pd.DataFrame(DatosLista, columns=['FechaHora', 'Miembro', 'Mensaje'])
df['FechaHora'] = pd.to_datetime(df['FechaHora'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['FechaHora', 'Miembro', 'Mensaje']).reset_index(drop=True)
df['Fecha'] = df['FechaHora'].dt.date
df['Hora'] = df['FechaHora'].dt.time

# Paso 3: Estadísticas de mensajes, multimedia, emojis y links
def ObtenerEmojis(Mensaje):
    emoji_lista = []
    data = regex.findall(r'\X', Mensaje)
    for caracter in data:
        if demoji.replace(caracter) != caracter:
            emoji_lista.append(caracter)
    return emoji_lista
total_mensajes = df.shape[0]
multimedia_mensajes = df[df['Mensaje'] == '<Multimedia omitido>'].shape[0]
df['Emojis'] = df['Mensaje'].apply(ObtenerEmojis)
emojis = sum(df['Emojis'].str.len())
url_patron = r'(https?://\S+)'
df['URLs'] = df.Mensaje.apply(lambda x: len(re.findall(url_patron, x)))
links = sum(df['URLs'])
encuestas = df[df['Mensaje'] == 'POLL:'].shape[0]
estadistica_dict = {'Tipo': ['Mensajes', 'Multimedia', 'Emojis', 'Links', 'Encuestas'],
        'Cantidad': [total_mensajes, multimedia_mensajes, emojis, links, encuestas]}
estadistica_df = pd.DataFrame(estadistica_dict, columns = ['Tipo', 'Cantidad'])
estadistica_df = estadistica_df.set_index('Tipo')

with st.expander('💡 ¡Cuánto hablamos! 🗣️', expanded=False):
    st.dataframe(estadistica_df)

emojis_lista = list([a for b in df.Emojis for a in b])
emoji_diccionario = dict(Counter(emojis_lista))
emoji_diccionario = sorted(emoji_diccionario.items(), key=lambda x: x[1], reverse=True)
emoji_df = pd.DataFrame(emoji_diccionario, columns=['Emoji', 'Cantidad'])
emoji_df = emoji_df.set_index('Emoji').head(10)
fig_emoji = px.pie(emoji_df, values='Cantidad', names=emoji_df.index, hole=.3, template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel2)
fig_emoji.update_traces(textposition='inside', textinfo='percent+label', textfont_size=20)
fig_emoji.update_layout(title={'text': 'Emojis que más usamos', 'y':0.96, 'x':0.5, 'xanchor': 'center'}, font=dict(size=17))

with st.expander('😍 Top 10 emojis que usamos para expresar nuestro cariño', expanded=False):
    st.dataframe(emoji_df)

with st.expander('😍 Así se ve nuestro amor en emojis', expanded=False):
    st.plotly_chart(fig_emoji)

# Miembros más activos
df_MiembrosActivos = df.groupby('Miembro')['Mensaje'].count().sort_values(ascending=False).to_frame()
df_MiembrosActivos.reset_index(inplace=True)
df_MiembrosActivos.index = np.arange(1, len(df_MiembrosActivos)+1)
df_MiembrosActivos['% Mensaje'] = (df_MiembrosActivos['Mensaje'] / df_MiembrosActivos['Mensaje'].sum()) * 100

# Estadísticas por miembro
multimedia_df = df[df['Mensaje'] == '<Multimedia omitido>']
mensajes_df = df.drop(multimedia_df.index)
mensajes_df['Letras'] = mensajes_df['Mensaje'].apply(lambda s : len(s))
mensajes_df['Palabras'] = mensajes_df['Mensaje'].apply(lambda s : len(s.split(' ')))
miembros = mensajes_df.Miembro.unique()
dictionario = {}
for i in range(len(miembros)):
    lista = []
    miembro_df= mensajes_df[mensajes_df['Miembro'] == miembros[i]]
    lista.append(miembro_df.shape[0])
    palabras_por_msj = (np.sum(miembro_df['Palabras']))/miembro_df.shape[0]
    lista.append(palabras_por_msj)
    multimedia = multimedia_df[multimedia_df['Miembro'] == miembros[i]].shape[0]
    lista.append(multimedia)
    emojis = sum(miembro_df['Emojis'].str.len())
    lista.append(emojis)
    links = sum(miembro_df['URLs'])
    lista.append(links)
    dictionario[miembros[i]] = lista
miembro_stats_df = pd.DataFrame.from_dict(dictionario)
estadísticas = ['Mensajes', 'Palabras por mensaje', 'Multimedia', 'Emojis', 'Links']
miembro_stats_df['Estadísticas'] = estadísticas
miembro_stats_df.set_index('Estadísticas', inplace=True)
miembro_stats_df = miembro_stats_df.T
miembro_stats_df['Mensajes'] = miembro_stats_df['Mensajes'].apply(int)
miembro_stats_df['Multimedia'] = miembro_stats_df['Multimedia'].apply(int)
miembro_stats_df['Emojis'] = miembro_stats_df['Emojis'].apply(int)
miembro_stats_df['Links'] = miembro_stats_df['Links'].apply(int)
miembro_stats_df = miembro_stats_df.sort_values(by=['Mensajes'], ascending=False)

with st.expander('👩‍❤️‍👨 ¿Quién envía más mensajes en nuestro chat?', expanded=False):
    st.dataframe(df_MiembrosActivos)

with st.expander('👩‍❤️‍👨 ¿Cómo se distribuyen nuestros mensajes?', expanded=False):
    st.dataframe(miembro_stats_df)

    

# Estadísticas de comportamiento del grupo
def create_range_hour(time_obj):
    if pd.isna(time_obj):
        return None
    start = time_obj.hour
    end = (start + 1) % 24
    return f'{start:02d} - {end:02d} h'
df['rangoHora'] = df['Hora'].apply(create_range_hour)
df['DiaSemana'] = df['FechaHora'].dt.day_name()
mapeo_dias_espanol = {'Monday': '1 Lunes','Tuesday': '2 Martes','Wednesday': '3 Miércoles','Thursday': '4 Jueves','Friday': '5 Viernes','Saturday': '6 Sábado','Sunday': '7 Domingo'}
df['DiaSemana'] = df['DiaSemana'].map(mapeo_dias_espanol)
df['# Mensajes por hora'] = 1
mensajes_hora = df.groupby('rangoHora').count().reset_index()
fig = px.line(mensajes_hora, x='rangoHora', y='# Mensajes por hora', color_discrete_sequence=['salmon'], template='plotly_dark')
fig.update_traces(mode='markers+lines', marker=dict(size=10))
fig.update_xaxes(title_text='Rango de hora', tickangle=30)
fig.update_yaxes(title_text='# Mensajes')
with st.expander('⏰ Nuestros horarios favoritos', expanded=False):
    st.plotly_chart(fig)
df['# Mensajes por día'] = 1
date_df = df.groupby('DiaSemana').count().reset_index()
fig = px.line(date_df, x='DiaSemana', y='# Mensajes por día', color_discrete_sequence=['salmon'], template='plotly_dark')
fig.update_traces(mode='markers+lines', marker=dict(size=10))
fig.update_xaxes(title_text='Día', tickangle=30)
fig.update_yaxes(title_text='# Mensajes')
with st.expander('📆 Nuestros días especiales', expanded=False):
    st.plotly_chart(fig)
date_df = df.groupby('Fecha')['# Mensajes por día'].sum().reset_index()
fig = px.line(date_df, x='Fecha', y='# Mensajes por día', color_discrete_sequence=['salmon'], template='plotly_dark')
fig.update_xaxes(title_text='Fecha', tickangle=45, nticks=35)
fig.update_yaxes(title_text='# Mensajes')
with st.expander('📈 Así crece nuestro amor!', expanded=False):
    st.plotly_chart(fig)
total_palabras = ' '
STOPWORDS.update(['que', 'qué', 'con', 'de', 'te', 'en', 'la', 'lo', 'le', 'el', 'las', 'los', 'les', 'por', 'es','son', 'se', 'para', 'un', 'una', 'chicos', 'su', 'si', 'chic','nos', 'ya', 'hay', 'esta','pero', 'del', 'mas', 'más', 'eso', 'este', 'como', 'así', 'todo', 'https','Multimedia','omitido','y', 'mi', 'o', 'q', 'yo', 'al'])
mask = np.array(Image.open('Resources/heart.jpg'))
for mensaje in mensajes_df['Mensaje'].values:
    palabras = str(mensaje).lower().split()
    for palabra in palabras:
        total_palabras = total_palabras + palabra + ' '
wordcloud = WordCloud(width = 800, height = 800, background_color ='black', stopwords = STOPWORDS,max_words=100, min_font_size = 5,mask = mask, colormap='OrRd',).generate(total_palabras)
with st.expander('❤️ ¡Lo que más decimos!', expanded=False):
    st.image(wordcloud.to_array(), caption='☁️ Nuestro word cloud', use_container_width=True)

st.markdown("<hr style='border:1px solid #d72660;'>", unsafe_allow_html=True)

# Mensaje final especial usando solo Streamlit
with st.container():
    st.subheader('💖 Gracias por cada momento, cada palabra y cada emoji')
    st.write('¡Por muchos meses más juntos!')
    st.write('Tu amor hace que cada chat sea especial.')
    st.markdown('**Te amo Paola** ❤️🥰')
