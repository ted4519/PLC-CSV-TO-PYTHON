import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import threading
import tkinter as tk
import time
import multiprocessing as mp
import signal
import seaborn as sns
import sys
import matplotlib.image as mpimg
import json
import os
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib import gridspec
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseEvent
from tkinter import colorchooser
from tkinter import messagebox
from tkinter import filedialog
from functools import partial
from matplotlib import rc


matplotlib.use('tkAgg')  # Tkinter 기반 백엔드 설정

# 한글 글꼴 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows


# Seaborn 테마 설정
sns.set_theme(style="whitegrid", font="Malgun Gothic")  # 스타일과 글꼴 설정

# 색상 저장/로드 관련 파일 경로
COLOR_FILE = "colors.json"

# 통합 CSV 파일 이름
output_file = "combined_temperature_data.csv"

# 전역 변수 선언
x_interval = "1초"  # 기본값
slider_offset_seconds = 0  # 슬라이더로 이동된 초
slider_active = False  # 슬라이더가 활성 상태인지 여부
num_temps = 16
slider_line = None  # 빨간 세로선을 관리하는 전역 변수


# X축 간격 옵션 선택에 따라 매핑
x_interval_map = {
    "1초": mdates.SecondLocator(interval=1),
    "5초": mdates.SecondLocator(interval=5),
    "10초": mdates.SecondLocator(interval=10),
    "1분": mdates.MinuteLocator(interval=1),
    "30분": mdates.MinuteLocator(interval=30),
    "60분": mdates.HourLocator(interval=1),
}


# 색상 정보를 파일에서 로드하는 함수
def load_colors():
    if os.path.exists(COLOR_FILE):
        with open(COLOR_FILE, "r") as f:
            return json.load(f)
    else:
        # 기본 색상
        return {
            f"TEMP{i + 1}": color
            for i, color in enumerate([
                "red", "blue", "green", "purple", "orange", "brown", "pink", "cyan", 
                "lime", "magenta", "teal", "gold", "coral", "indigo", "salmon", "navy"
            ])
        }

# 색상 정보를 파일에 저장하는 함수
def save_colors(colors):
    with open(COLOR_FILE, "w") as f:
        json.dump(colors, f)

# 현재 색상 로드
colors = load_colors()




# 데이터 생성 및 저장 함수
def save_temperature_data_real_time(file_name):
    # 초기 CSV 파일 생성
    columns = ["TIME"] + [f"TEMP{i}" for i in range(1, num_temps + 1)]
    df = pd.DataFrame(columns=columns)  # 빈 데이터프레임 생성
    df.to_csv(file_name, index=False)  # 초기 헤더 작성

    while True:
        # 현재 시간 생성
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # TEMP 데이터 생성
        temp_values = [np.round(np.random.uniform(0, 1000), 1) for _ in range(num_temps)]

        # 새 행 추가
        new_row = [current_time] + temp_values
        df.loc[len(df)] = new_row

        # 데이터를 파일에 저장 (추가 모드)
        df.tail(1).to_csv(file_name, index=False, header=False, mode='a')

        time.sleep(1)  # 1초 대기

# 데이터를 저장하는 스레드 시작
thread = threading.Thread(target=save_temperature_data_real_time, args=(output_file,), daemon=True)
thread.start()




# 레이아웃 설정: 두 그래프와 버튼 영역
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 2, width_ratios=[5.5, 1.5], height_ratios=[0.8, 1])

# 첫 번째 그래프 영역
ax_graph = fig.add_subplot(gs[0, 0])

# 두 번째 그래프 영역
ax_second_graph = fig.add_subplot(gs[1, 0])

# 버튼 영역 설정
ax_button_area = fig.add_subplot(gs[:, 1])
ax_button_area.axis("off")

# 버튼 설정
buttons = []
button_texts = []
button_data_map = {}
button_states = {}
total_buttons = 16
button_height = 0.036
button_width = 0.05   # 버튼 너비를 약간 줄임
vertical_spacing = 0.015  # 버튼 간격을 약간 줄임
start_y = 0.91  # 버튼 시작 위치 (y좌표)

# 버튼과 두 텍스트 박스를 나란히 표시하는 영역 생성
slider_texts = []  # 슬라이더 값 텍스트를 저장할 리스트


for i in range(total_buttons):
    y = start_y - i * (button_height + vertical_spacing)
    
    # 버튼 생성
    button_ax = plt.axes([0.85, y, button_width, button_height])  # 너비와 높이를 조정
    btn = Button(button_ax, f"TEMP{i + 1}")
    btn.color = colors[f"TEMP{i + 1}"]
    btn.hovercolor = colors[f"TEMP{i + 1}"]
    btn.label.set_color("white")
    btn.label.set_fontweight("bold")
    btn.label.set_fontsize(9)
    buttons.append(btn)
    
    # 실시간 값 텍스트 박스 (버튼 바로 오른쪽에 배치)
    text_ax = plt.axes([0.91, y, button_width * 0.7, button_height])
    text = text_ax.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=10, 
        transform=text_ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.2", edgecolor="black", facecolor="white")
    )
    text_ax.axis("off")
    button_texts.append(text)
    
    # 슬라이더 값 텍스트 박스 (실시간 값 옆에 배치)
    slider_text_ax = plt.axes([0.96, y, button_width * 0.7, button_height])
    slider_text = slider_text_ax.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=10, 
        transform=slider_text_ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.2", edgecolor="black", facecolor="white")
    )
    slider_text_ax.axis("off")
    slider_texts.append(slider_text)
    
    # 버튼과 TEMP 컬럼 매핑
    button_data_map[btn] = f"TEMP{i + 1}"  # TEMP1, TEMP2 등 컬럼명으로 매핑
    button_states[btn] = False


# Save 버튼 생성
save_button_ax = plt.axes([0.85, start_y - total_buttons * (button_height + vertical_spacing) - 0.01, button_width, button_height])  
save_button = Button(save_button_ax, 'Save Data')


# 저장 상태를 나타내는 변수
save_start_time = None  # 저장 시작 시점
save_active = False  # 저장 활성화 상태

def toggle_save_data(event):
    """
    데이터를 저장 시작/정지하는 기능
    """
    global save_start_time, save_active

    if not save_active:
        # 저장 시작
        save_start_time = datetime.datetime.now()
        save_active = True
        save_button.label.set_text('Stop Save')  # 버튼 텍스트 변경
    else:
        # 저장 종료
        save_end_time = datetime.datetime.now()
        save_active = False
        save_button.label.set_text('Save Data')  # 버튼 텍스트 변경

        # 저장된 데이터를 CSV로 출력
        try:
            df = pd.read_csv(output_file)
            df['TIME'] = pd.to_datetime(df['TIME'])  # TIME 열 변환

            # 시작 및 종료 시간에 해당하는 데이터 필터링
            filtered_data = df[(df['TIME'] >= save_start_time) & (df['TIME'] <= save_end_time)]

            if not filtered_data.empty:
                # 파일 저장
                output_file_name = f"saved_data_{save_start_time.strftime('%Y%m%d_%H%M%S')}_to_{save_end_time.strftime('%Y%m%d_%H%M%S')}.csv"
                filtered_data.to_csv(output_file_name, index=False)

                # 저장 완료 안내 창 띄우기
                messagebox.showinfo("Save Complete", f"Data saved successfully to:\n{output_file_name}")
            else:
                messagebox.showwarning("No Data", "No data available to save in the selected range.")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving data:\n{e}")


# Save 버튼 이벤트 연결
save_button.on_clicked(toggle_save_data)





# 정지 및 플레이 버튼 설정
toggle_button_ax = plt.axes([0.85, 0.02, 0.07, button_height])
play_image = mpimg.imread("play_button.png")
stop_image = mpimg.imread("stop_button.png")
current_image = "stop"
toggle_button_display = toggle_button_ax.imshow(stop_image)
toggle_button_ax.axis("off")
animation_running = True


def toggle_animation(event):
    global ani, animation_running, current_image, toggle_button_display

    if toggle_button_ax.contains(event)[0]:
        if animation_running:
            ani.event_source.stop()
            animation_running = False
            current_image = "play"
            toggle_button_display.set_data(play_image)
        else:
            ani.event_source.start()
            animation_running = True
            current_image = "stop"
            toggle_button_display.set_data(stop_image)

        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', toggle_animation)


def create_button_handler(temp_label, color, btn, idx):
    def handler(event):
        global ani, active_data, button_states
        if event.button != 1:  # 왼쪽 클릭만 처리
            return

        if button_states[btn]:
            # TEMP 데이터 비활성화
            active_data.pop(temp_label, None)
            button_states[btn] = False
            button_texts[idx].set_text("")  # 버튼 텍스트 초기화
        else:
            # TEMP 데이터 활성화
            active_data[temp_label] = (color, temp_label)  # 컬럼명 추가
            button_states[btn] = True

        if ani is None:  # 애니메이션 시작
            ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
        plt.draw()

    return handler




def open_color_picker(event, button, idx):
    global colors

    root = tk.Tk()
    root.withdraw()

    chosen_color = colorchooser.askcolor(title=f"Select Color for {button.label.get_text()}")[1]
    if chosen_color:
        temp_key = f"TEMP{idx + 1}"
        colors[temp_key] = chosen_color
        button.color = chosen_color
        button.hovercolor = chosen_color
        button.label.set_color("white")
        save_colors(colors)  
        plt.draw()
    root.destroy()

def connect_right_click_event():
    for i, button in enumerate(buttons):
        button_ax = button.ax
        def callback(event, btn=button, idx=i):
            if event.button == 3 and event.inaxes == btn.ax:
                open_color_picker(event, btn, idx)
        button_ax.figure.canvas.mpl_connect("button_press_event", callback)

ani = None
active_data = {}

for i, button in enumerate(buttons):
    temp_label = f"TEMP{i + 1}"  # TEMP1, TEMP2, ...
    color = colors[temp_label]  # 버튼에 설정된 색상 가져오기
    handler = lambda event, temp_label=temp_label, color=color, button=button, idx=i: create_button_handler(temp_label, color, button, idx)(event)
    button.on_clicked(handler)  # 핸들러 등록




time_button_ax = plt.axes([0.005, 0.13, 0.04, 0.02])
time_button = Button(time_button_ax, 'Set Time')

# 추가: 전역 변수 선언
selected_temp = None
start_time = None
end_time = None


def open_time_setting(event):
    """
    시작 시간과 종료 시간을 설정할 수 있는 팝업 창
    """
    global popup, temp_var, x_interval_var  
    global start_hour_spinbox, start_minute_spinbox, start_second_spinbox
    global end_hour_spinbox, end_minute_spinbox, end_second_spinbox

    # Spinbox 초기화 강제 확인 함수
    def ensure_spinboxes_initialized():
        global start_hour_spinbox, start_minute_spinbox, start_second_spinbox
        global end_hour_spinbox, end_minute_spinbox, end_second_spinbox

        try:
            if not (start_hour_spinbox and start_minute_spinbox and start_second_spinbox):
                raise ValueError("Start Spinboxes are not initialized.")
            if not (end_hour_spinbox and end_minute_spinbox and end_second_spinbox):
                raise ValueError("End Spinboxes are not initialized.")


        except ValueError as e:
            tk.messagebox.showerror("Error", "Spinbox objects are not properly initialized.")

    def update_selected_label(*args):
        selected_label.config(text=f"Selected: {x_interval_var.get()}")

    def submit_time():
        global selected_temp, start_time, end_time, x_interval_var, x_interval

        # TEMP 값 확인
        if selected_temp is None or selected_temp.strip() == "":
            selected_temp = temp_var.get()  # TEMP 값을 강제로 가져옴

        # TEMP 선택 여부 확인
        if selected_temp is None or selected_temp.strip() == "":
            tk.messagebox.showerror("Error", "Please select a TEMP value.")
            return
        
        
        try:
            # Spinbox 값 가져오기
            start_hour = start_hour_spinbox.get().strip()
            start_minute = start_minute_spinbox.get().strip()
            start_second = start_second_spinbox.get().strip()
            end_hour = end_hour_spinbox.get().strip()
            end_minute = end_minute_spinbox.get().strip()
            end_second = end_second_spinbox.get().strip()

            
            # 값 비어 있는 경우 오류 처리
            if not all([start_hour, start_minute, start_second, end_hour, end_minute, end_second]):
                tk.messagebox.showerror("Error", "Please enter valid time values.")
                return

            # Spinbox 값을 정수로 변환
            try:
                start_hour = int(start_hour)
                start_minute = int(start_minute)
                start_second = int(start_second)
                end_hour = int(end_hour)
                end_minute = int(end_minute)
                end_second = int(end_second)
            except ValueError as e:
                tk.messagebox.showerror("Error", "Spinbox values must be valid integers.")
                return

            # 시간 객체 생성 및 전역 변수 업데이트
            try:
                start_time = datetime.time(start_hour, start_minute, start_second)
                end_time = datetime.time(end_hour, end_minute, end_second)

                
            except Exception as e:
                tk.messagebox.showerror("Error", "Invalid time values provided.")
                return

            # 유효성 검사
            if start_time >= end_time:
                tk.messagebox.showerror("Invalid Time", "Start time must be earlier than End time!")
                return


            if not selected_temp:
                tk.messagebox.showerror("Error", "Please select a TEMP value.")
                return

            # 팝업 닫기
            popup.destroy()

        except Exception as e:
            tk.messagebox.showerror("Error", f"An unexpected error occurred: {e}")




    def save_time_range_data():
        global selected_temp, start_time, end_time

        # TEMP 값 강제 설정
        if selected_temp is None:
            selected_temp = temp_var.get()  # TEMP 메뉴에서 선택한 값 가져오기

        # 시간 범위 설정 확인
        submit_time()  # "Submit" 기능을 저장 시에도 호출

        # TEMP와 시간 값이 설정되지 않았다면 오류 반환
        if selected_temp is None or start_time is None or end_time is None:
            tk.messagebox.showerror("Error", "Please set a valid TEMP and time range!")
            return

        try:
            # 시간 설정 유효성 검사
            if start_time >= end_time:
                tk.messagebox.showerror("Error", "Start time must be earlier than End time!")
                return

            # 통합 파일에서 데이터 읽기
            df = pd.read_csv(output_file)
            df['TIME'] = pd.to_datetime(df['TIME'])  # 시간 데이터 변환

            # 시간 범위 필터링
            filtered_data = df[
                (df['TIME'].dt.time >= start_time) &
                (df['TIME'].dt.time <= end_time)
            ][['TIME', selected_temp]].dropna()

            # 데이터 저장
            if not filtered_data.empty:
                output_file_name = f"{selected_temp}_data_{start_time.strftime('%H%M%S')}_{end_time.strftime('%H%M%S')}.csv"
                filtered_data.to_csv(output_file_name, index=False)
                tk.messagebox.showinfo("Success", f"Data saved to {output_file_name}")
            else:
                tk.messagebox.showwarning("No Data", "No data available in the selected range!")

        except Exception as e:
            tk.messagebox.showerror("Error", f"Error saving data: {e}")



    # TEMP 초기값 설정
    selected_temp = "TEMP1"  # TEMP1을 기본 선택값으로 설정
    temp_var = tk.StringVar(value="TEMP1")  # TEMP1을 초기값으로 지정


    def update_temp_label(*args):
        global selected_temp
        selected_temp = temp_var.get()
        temp_label.config(text=f"Selected: {temp_var.get()}")

    # X축 간격 설정 함수
    def update_x_interval(*args):
        global x_interval
        x_interval = x_interval_var.get()

    popup = tk.Tk()
    popup.title("Set Time")
    popup.geometry("650x520+750+750")  # X=750, Y=750 위치에 표시

    # TEMP 선택 영역
    temp_frame = tk.LabelFrame(popup, text="Select TEMP", font=("Helvetica", 14, "bold"), padx=10, pady=10)
    temp_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")  # 간격 좁힘

    temp_var = tk.StringVar(value="TEMP1")
    temp_var.trace("w", update_temp_label)  
    temp_options = [f"TEMP{i}" for i in range(1, 17)]

    temp_label = tk.Label(temp_frame, text=f"Selected: {temp_var.get()}", font=("Helvetica", 12))
    temp_label.pack(pady=5)

    temp_menu = tk.OptionMenu(temp_frame, temp_var, *temp_options)
    temp_menu.config(font=("Helvetica", 12), width=10)
    temp_menu.pack(pady=10)

    # X축 간격 선택
    x_interval_frame = tk.LabelFrame(popup, text="Select X-Axis Interval", font=("Helvetica", 14, "bold"), padx=10, pady=10)
    x_interval_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nw")  # TEMP와 간격 좁힘

    x_interval_var = tk.StringVar(value="1초")  
    x_interval_var.trace("w", update_x_interval)  # 변경 시 호출

    selected_label = tk.Label(x_interval_frame, text=f"Selected: {x_interval_var.get()}", font=("Helvetica", 12))
    selected_label.pack(pady=5)  # 선택된 간격 표시

    x_interval_menu = tk.OptionMenu(x_interval_frame, x_interval_var, "1초", "5초", "10초", "1분", "30분", "60분")
    x_interval_menu.config(font=("Helvetica", 12), width=10)
    x_interval_menu.pack(pady=10)

    x_interval_var.trace("w", update_selected_label)

    # 시간 설정 UI
    time_frame = tk.LabelFrame(popup, text="Set Time", font=("Helvetica", 14, "bold"), padx=10, pady=10)
    time_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="ne")

    start_frame = tk.LabelFrame(time_frame, text="Start Time", font=("Helvetica", 12), padx=10, pady=10)
    start_frame.grid(row=0, column=0, padx=10, pady=10)

    start_hour_spinbox = tk.Spinbox(start_frame, from_=0, to=23, font=("Helvetica", 12), width=5)
    start_hour_spinbox.delete(0, "end")
    start_hour_spinbox.insert(0, "0")  # 기본값 설정
    start_hour_spinbox.grid(row=0, column=1)
    tk.Label(start_frame, text="Hour", font=("Helvetica", 12)).grid(row=0, column=0)

    start_minute_spinbox = tk.Spinbox(start_frame, from_=0, to=59, font=("Helvetica", 12), width=5)
    start_minute_spinbox.delete(0, "end")
    start_minute_spinbox.insert(0, "0")
    start_minute_spinbox.grid(row=1, column=1)
    tk.Label(start_frame, text="Minute", font=("Helvetica", 12)).grid(row=1, column=0)

    start_second_spinbox = tk.Spinbox(start_frame, from_=0, to=59, font=("Helvetica", 12), width=5)
    start_second_spinbox.delete(0, "end")
    start_second_spinbox.insert(0, "0")
    start_second_spinbox.grid(row=2, column=1)
    tk.Label(start_frame, text="Second", font=("Helvetica", 12)).grid(row=2, column=0)

    end_frame = tk.LabelFrame(time_frame, text="End Time", font=("Helvetica", 12), padx=10, pady=10)
    end_frame.grid(row=1, column=0, padx=10, pady=10)

    end_hour_spinbox = tk.Spinbox(end_frame, from_=0, to=23, font=("Helvetica", 12), width=5)
    end_hour_spinbox.delete(0, "end")
    end_hour_spinbox.insert(0, "0")
    end_hour_spinbox.grid(row=0, column=1)
    tk.Label(end_frame, text="Hour", font=("Helvetica", 12)).grid(row=0, column=0)

    end_minute_spinbox = tk.Spinbox(end_frame, from_=0, to=59, font=("Helvetica", 12), width=5)
    end_minute_spinbox.delete(0, "end")
    end_minute_spinbox.insert(0, "0")
    end_minute_spinbox.grid(row=1, column=1)
    tk.Label(end_frame, text="Minute", font=("Helvetica", 12)).grid(row=1, column=0)

    end_second_spinbox = tk.Spinbox(end_frame, from_=0, to=59, font=("Helvetica", 12), width=5)
    end_second_spinbox.delete(0, "end")
    end_second_spinbox.insert(0, "0")
    end_second_spinbox.grid(row=2, column=1)
    tk.Label(end_frame, text="Second", font=("Helvetica", 12)).grid(row=2, column=0)


    # Submit 버튼
    tk.Button(popup, text="Submit", command=submit_time, font=("Helvetica", 12)).grid(row=2, column=0, columnspan=2, pady=20)

    # Save 버튼 추가
    tk.Button(popup, text="Save Data", command=save_time_range_data, font=("Helvetica", 12)).grid(row=2, column=1, pady=20, padx=10)

    popup.mainloop()

# X축 간격 업데이트 함수
def update_x_interval(*args):
    global x_interval
    x_interval = x_interval_var.get()



def update(frame):
    """
    애니메이션 업데이트 함수.
    여러 TEMP를 선택할 경우 그래프를 중첩하여 표시.
    """
    global active_data, selected_temp, start_time, end_time, x_interval, slider_offset_seconds

    # 첫 번째 그래프 업데이트
    ax_graph.clear()
    current_time = datetime.datetime.now()
    start_window = current_time - datetime.timedelta(seconds=60)

    data_added = False

    try:
        # 통합 파일 읽기
        df = pd.read_csv(output_file)
        df['TIME'] = pd.to_datetime(df['TIME'])  # TIME 열 변환

        for temp_label, (color, temp_column) in active_data.items():
            # 선택된 TEMP 데이터만 필터링
            recent_data = df[(df['TIME'] >= start_window)][['TIME', temp_column]].dropna()

            if recent_data.empty:
                button_texts[buttons.index(next(b for b, t in button_data_map.items() if t == temp_label))].set_text("")
                continue

            latest_row = recent_data.iloc[-1]
            button_texts[buttons.index(next(b for b, t in button_data_map.items() if t == temp_label))].set_text(
                f"{latest_row['TIME'].strftime('%H:%M:%S')}\n{latest_row[temp_column]}°C"
            )

            # 그래프에 TEMP 데이터 추가
            sns.lineplot(
                x=recent_data['TIME'], 
                y=recent_data[temp_column], 
                label=temp_column, 
                ax=ax_graph, 
                color=color, 
                linestyle='-'
            )
            # X축 레이블 제거
            ax_graph.set_xlabel("") 
            data_added = True

    except Exception as e:
        print(f"Error updating first graph: {e}")

    if data_added:
        ax_graph.legend(fontsize=10)

    ax_graph.set_xlim(start_window, current_time)
    ax_graph.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
    ax_graph.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax_graph.set_title("Real-Time Temperature Changes", fontsize=16)
    ax_graph.set_ylabel("Temperature (°C)", fontsize=14)
    ax_graph.grid(True)

    plt.setp(ax_graph.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 두 번째 그래프 업데이트
    if start_time is not None and end_time is not None:
        ax_second_graph.cla()  # 기존 그래프 초기화

        try:
            for button, temp_label in button_data_map.items():
                if button_states[button]:  # TEMP가 활성 상태인 경우에만 그래프 그리기
                    color = colors[temp_label]
                    temp_column = temp_label

                    # 시간 범위 설정
                    new_start = (datetime.datetime.combine(datetime.date.today(), start_time)).time()
                    new_end = (datetime.datetime.combine(datetime.date.today(), end_time)).time()

                    filtered_data = df[
                        (df['TIME'].dt.time >= new_start) & 
                        (df['TIME'].dt.time <= new_end)
                    ][['TIME', temp_column]].dropna()

                    if not filtered_data.empty:
                        # 두 번째 그래프에 TEMP 데이터 추가 (Seaborn 스타일 적용)
                        sns.lineplot(
                            x=filtered_data['TIME'], 
                            y=filtered_data[temp_column], 
                            label=temp_column, 
                            ax=ax_second_graph, 
                            color=color, 
                            linestyle='-'
                        )
                        # X축 레이블 제거
                        ax_second_graph.set_xlabel("")


            # X축 간격 적용
            if x_interval in x_interval_map:
                ax_second_graph.xaxis.set_major_locator(x_interval_map[x_interval])
                ax_second_graph.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

            ax_second_graph.legend(fontsize=10)
            ax_second_graph.set_title("Temperature in Selected Time Range", fontsize=16)
            ax_second_graph.set_ylabel("Temperature (°C)", fontsize=14)
            ax_second_graph.grid(True)
            plt.setp(ax_second_graph.xaxis.get_majorticklabels(), rotation=45, ha="right")

        except Exception as e:
            print(f"Error updating second graph: {e}")

        # 두 번째 그래프에서 빨간 선 유지
        if slider_line:
            ax_second_graph.add_line(slider_line)  # 그래프 갱신 시 빨간 선 다시 추가
    plt.draw()





# 슬라이더 추가
slider_ax = plt.axes([0.05, 0.01, 0.79, 0.03], facecolor='lightgoldenrodyellow')  # bottom 값을 더 낮게 조정
time_slider = Slider(slider_ax, 'SCROLL', 0, 60, valinit=0, valstep=1)  # 슬라이더 설정


updating_slider = False  # 재귀 호출 방지 플래그

def update_slider(val):
    """
    슬라이더 값을 그래프의 X축 눈금에 맞춰 스냅하고 TEMP 값을 갱신.
    """
    global slider_line, updating_slider

    if updating_slider:  # 재귀 호출 방지
        return

    try:
        updating_slider = True

        # 그래프 X축 눈금 가져오기
        x_ticks = ax_second_graph.get_xticks()
        if len(x_ticks) < 2:
            print("No sufficient ticks on X-axis.")
            return

        # 슬라이더 값을 그래프 X축 데이터 범위에 매핑
        graph_min, graph_max = ax_second_graph.get_xlim()
        slider_mapped_val = graph_min + (val - time_slider.valmin) / (time_slider.valmax - time_slider.valmin) * (graph_max - graph_min)

        # 슬라이더 값을 가장 가까운 X축 눈금으로 스냅
        closest_tick = min(x_ticks, key=lambda x: abs(x - slider_mapped_val))

        # 빨간선 위치 업데이트
        if slider_line:
            slider_line.set_xdata([closest_tick])
        else:
            slider_line = ax_second_graph.axvline(
                x=closest_tick, color='red', linestyle='--', linewidth=1.5
            )

        # 슬라이더 값을 시간으로 변환
        slider_time = matplotlib.dates.num2date(closest_tick)
        slider_time = pd.Timestamp(slider_time).tz_localize(None)

        # CSV 데이터에서 가장 가까운 시간 찾기
        df = pd.read_csv(output_file)
        df['TIME'] = pd.to_datetime(df['TIME']).dt.tz_localize(None)

        closest_row = df.iloc[(df['TIME'] - slider_time).abs().argsort()[:1]]

        # TEMP 값 및 시간 표시
        if not closest_row.empty:
            display_time = closest_row['TIME'].iloc[0].strftime("%H:%M:%S")

            for idx, temp_label in enumerate(button_data_map.values()):
                if button_states[buttons[idx]]:
                    temp_value = closest_row[temp_label].iloc[0]
                    slider_texts[idx].set_text(f"{display_time}\n{temp_value:.1f}°C")
                else:
                    slider_texts[idx].set_text("")
        else:
            print("No closest row found for slider time.")

        plt.draw()

    except Exception as e:
        print(f"Error updating slider: {e}")

    finally:
        updating_slider = False  # 슬라이더 업데이트 완료





def initialize_slider():
    """
    슬라이더를 그래프의 X축 데이터 범위에 맞춰 초기화.
    """
    df = pd.read_csv(output_file)
    df['TIME'] = pd.to_datetime(df['TIME'])

    if not df.empty:
        time_min = matplotlib.dates.date2num(df['TIME'].min())
        time_max = matplotlib.dates.date2num(df['TIME'].max())

        # 그래프 X축 설정
        ax_second_graph.set_xlim(time_min, time_max)
        ax_second_graph.figure.canvas.draw()

        # 슬라이더 범위 설정
        time_slider.valmin = time_min
        time_slider.valmax = time_max
        time_slider.valstep = (time_max - time_min) / 100  # 세분화된 스텝
        time_slider.set_val(time_min)  # 슬라이더 초기값

        print("Slider initialized with:")
        print(f"valmin={time_slider.valmin}, valmax={time_slider.valmax}, valstep={time_slider.valstep}")
        print(f"Graph X-axis range: {time_min} to {time_max}")





def rescale_graph_xaxis():
    """
    그래프의 X축을 데이터 범위에 맞춰 자동 조정.
    """
    df = pd.read_csv(output_file)
    df['TIME'] = pd.to_datetime(df['TIME'])
    
    # 데이터가 비어있지 않으면 X축 범위 재설정
    if not df.empty:
        time_min = matplotlib.dates.date2num(df['TIME'].min())
        time_max = matplotlib.dates.date2num(df['TIME'].max())

        # X축 재설정
        ax_second_graph.set_xlim(time_min, time_max)
        print(f"Graph X-axis updated to: {time_min} - {time_max}")

    initialize_slider()  # 슬라이더 다시 초기화



# 슬라이더 이벤트 연결
time_slider.on_changed(update_slider)


# 파일 불러오기 버튼 설정
file_button_ax = plt.axes([0.005, 0.1, 0.04, 0.02])
file_button = Button(file_button_ax, 'Load File')

def load_file(event):
    """
    파일을 선택하여 데이터를 불러오고 두 번째 그래프에 출력합니다.
    """
    global ax_second_graph

    # 파일 선택 다이얼로그 열기
    file_path = filedialog.askopenfilename(
        title="Select a CSV File",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    
    if not file_path:
        return  # 사용자가 파일 선택을 취소한 경우

    try:
        # 파일 읽기
        loaded_df = pd.read_csv(file_path)
        loaded_df['TIME'] = pd.to_datetime(loaded_df['TIME'])

        # 데이터 출력: TEMP1만 표시 (다른 TEMP 데이터도 출력하려면 확장 가능)
        ax_second_graph.clear()
        for col in loaded_df.columns:
            if col.startswith("TEMP"):
                # Seaborn 스타일로 TEMP 데이터 추가
                sns.lineplot(
                    x=loaded_df['TIME'], 
                    y=loaded_df[col], 
                    label=col, 
                    ax=ax_second_graph, 
                    linestyle='-'
                )
                # X축 레이블 제거
                ax_second_graph.set_xlabel("")

        
        ax_second_graph.legend()
        ax_second_graph.set_title(f"Data from {file_path}")
        ax_second_graph.set_xlabel("Time")
        ax_second_graph.set_ylabel("Temperature (°C)")
        ax_second_graph.grid(True)
        plt.setp(ax_second_graph.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 두 번째 그래프를 새로고침
        plt.draw()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while loading the file:\n{e}")

# 파일 불러오기 버튼 이벤트 연결
file_button.on_clicked(load_file)






plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=1.15, hspace=0.3)

connect_right_click_event()
# 슬라이더 비활성화 함수 추가
def deactivate_slider(event):
    """
    슬라이더가 멈춘 후 상태를 비활성화.
    """
    global slider_active
    slider_active = False

# 슬라이더 비활성화 이벤트 연결
fig.canvas.mpl_connect('button_release_event', deactivate_slider)

time_button.on_clicked(open_time_setting)

def set_window_icon(icon_path):
    try:
        root = matplotlib.pyplot.get_current_fig_manager().window
        if isinstance(root, tk.Tk):
            root.iconbitmap(icon_path)
        elif hasattr(root, 'tk'):
            root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file=icon_path))
    except Exception as e:
        print(f"Error setting icon: {e}")

icon_file = r"C:\Users\ted45\OneDrive\바탕 화면\PLC\icon.ico"
set_window_icon(icon_file)

plt.gcf().canvas.manager.set_window_title("Temperature Monitoring System")

plt.show()