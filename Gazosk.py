import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
import os
import configparser
import platform
import locale
import sys

APP_NAME = "画像さがす君・Local AI Image Search"
APP_VERSION = "1.0.1"
CONFIG_FILE = "gui_config.ini"
README_FILE = "README.md"
# 実行ファイル名はOSによって変える想定
EXECUTABLE_NAME = "LocalAIImageSearch.exe" if platform.system() == "Windows" else "LocalAIImageSearch_Mac"

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_NAME)
        self.geometry("700x550")
        try:
            # OSごとにアイコンファイルを切り替え
            icon_file = "gazosk.icns" if platform.system() == "Darwin" else "gazosk.ico"
            icon_path = resource_path(icon_file)
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            else:
                # 存在しない場合は何もしない（エラーも出さない）
                print(f"Info: アイコンファイル '{icon_file}' が見つかりませんでした。")
        except tk.TclError:
            # .iconbitmapがTcl/Tkでサポートされていない場合のエラー
            print("Info: この環境ではウィンドウアイコンの設定がサポートされていません。")


        # --- State Variables ---
        self.generate_process = None
        self.search_process = None
        self.target_path = ctk.StringVar()

        # --- Load Config ---
        self.load_config()

        # --- UI Setup ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.tabview.add("Generate")
        self.tabview.add("Search")
        self.tabview.add("Help")

        self.create_generate_tab(self.tabview.tab("Generate"))
        self.create_search_tab(self.tabview.tab("Search"))
        self.create_help_tab(self.tabview.tab("Help"))

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_generate_tab(self, tab):
        tab.grid_columnconfigure(1, weight=1)

        # 1. Target Path
        ctk.CTkLabel(tab, text="処理対象パス:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        path_entry = ctk.CTkEntry(tab, textvariable=self.target_path, width=300)
        path_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        browse_button = ctk.CTkButton(tab, text="参照...", command=self.browse_directory)
        browse_button.grid(row=0, column=2, padx=10, pady=10)

        # 2. Number of items to generate
        ctk.CTkLabel(tab, text="生成件数:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.generate_count_entry = ctk.CTkEntry(tab, placeholder_text="未指定時は1000件/日, 0で無制限")
        self.generate_count_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # Description text
        desc_text = "指定がないときは1日1000リクエストの範囲内、0のときは制限ナシなどの説明を画面に表示"
        ctk.CTkLabel(tab, text=desc_text, wraplength=400, justify="left").grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="w")

        # 3. Action Buttons
        button_frame = ctk.CTkFrame(tab)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        self.generate_button = ctk.CTkButton(button_frame, text="生成 開始", command=self.start_generation)
        self.generate_button.pack(side="left", padx=10)
        self.generate_stop_button = ctk.CTkButton(button_frame, text="停止", command=self.stop_generation, state="disabled")
        self.generate_stop_button.pack(side="left", padx=10)

        # 4. Output Console
        ctk.CTkLabel(tab, text="実行ログ:").grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")
        self.generate_output_text = ctk.CTkTextbox(tab, height=150)
        self.generate_output_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        tab.grid_rowconfigure(5, weight=1)


    def create_search_tab(self, tab):
        tab.grid_columnconfigure(1, weight=1)

        # 0. Target Path (Shared with Generate tab)
        ctk.CTkLabel(tab, text="処理対象パス:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        path_entry = ctk.CTkEntry(tab, textvariable=self.target_path, width=300)
        path_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        browse_button = ctk.CTkButton(tab, text="参照...", command=self.browse_directory)
        browse_button.grid(row=0, column=2, padx=10, pady=10)

        # 1. Query
        ctk.CTkLabel(tab, text="クエリ(検索語):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.query_entry = ctk.CTkEntry(tab, width=300)
        self.query_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky="ew")

        # 2. Options
        options_frame = ctk.CTkFrame(tab)
        options_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        self.no_word_var = ctk.BooleanVar()
        ctk.CTkCheckBox(options_frame, text="テキストマッチを行わない (-noword)", variable=self.no_word_var).pack(anchor="w", padx=10, pady=5)

        self.no_exword_var = ctk.BooleanVar()
        ctk.CTkCheckBox(options_frame, text="キーワードを分解しない (-noexword)", variable=self.no_exword_var).pack(anchor="w", padx=10, pady=5)

        self.no_cos_var = ctk.BooleanVar()
        ctk.CTkCheckBox(options_frame, text="コサイン類似度評価を行わない (-nocos)", variable=self.no_cos_var).pack(anchor="w", padx=10, pady=5)

        # Result count
        n_frame = ctk.CTkFrame(options_frame)
        n_frame.pack(anchor="w", padx=5, pady=5, fill="x")
        ctk.CTkLabel(n_frame, text="-n:").pack(side="left", padx=(5,0))
        self.n_entry = ctk.CTkEntry(n_frame, width=80)
        self.n_entry.pack(side="left", padx=5)
        ctk.CTkLabel(n_frame, text="(結果の件数を指定, 0で無制限)").pack(side="left", padx=5)

        # Time range
        t_frame = ctk.CTkFrame(options_frame)
        t_frame.pack(anchor="w", padx=5, pady=5, fill="x")
        ctk.CTkLabel(t_frame, text="-t:").pack(side="left", padx=(5,0))
        self.t_entry = ctk.CTkEntry(t_frame, width=150)
        self.t_entry.pack(side="left", padx=5)
        ctk.CTkLabel(t_frame, text="(画像の作成日でフィルタ)").pack(side="left", padx=5)

        # 3. Action Buttons
        button_frame = ctk.CTkFrame(tab)
        button_frame.grid(row=3, column=0, columnspan=3, pady=20)
        self.search_button = ctk.CTkButton(button_frame, text="検索 実行", command=self.start_search)
        self.search_button.pack(side="left", padx=10)
        self.search_stop_button = ctk.CTkButton(button_frame, text="停止", command=self.stop_search, state="disabled")
        self.search_stop_button.pack(side="left", padx=10)

        # 4. Output Console
        ctk.CTkLabel(tab, text="実行ログ:").grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")
        self.search_output_text = ctk.CTkTextbox(tab, height=150)
        self.search_output_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        tab.grid_rowconfigure(5, weight=1)

    def create_help_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        help_frame = ctk.CTkFrame(tab)
        help_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        load_readme_button = ctk.CTkButton(help_frame, text="README.md を表示", command=self.load_readme)
        load_readme_button.pack(side="left", padx=10, pady=10)
        
        # 右寄せでバージョンとクレジットを表示するためのフレーム
        info_frame = ctk.CTkFrame(help_frame)
        info_frame.configure(fg_color="transparent") # 背景を透過
        info_frame.pack(side="right", padx=10, pady=5)

        version_label = ctk.CTkLabel(info_frame, text=f"Version: {APP_VERSION}")
        version_label.pack(anchor="e")
        
        credit_label = ctk.CTkLabel(info_frame, text="(c) 2025 / Satoshi Endo @hortense667")
        credit_label.pack(anchor="e")

        self.readme_text = ctk.CTkTextbox(tab)
        self.readme_text.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.readme_text.insert("0.0", "「README.md を表示」ボタンを押してください。")
        self.readme_text.configure(state="disabled")

    def _read_process_output(self, process, text_widget):
        """プロセスの出力をリアルタイムで読み取り、テキストウィジェットに表示する"""
        try:
            for line in iter(process.stdout.readline, ''):
                # GUIの更新はメインスレッドで行う
                self.after(0, lambda l=line: text_widget.insert(tk.END, l))
                text_widget.see(tk.END) # 自動スクロール
            process.stdout.close()
            process.wait()
        except Exception as e:
            self.after(0, lambda: text_widget.insert(tk.END, f"\n--- ログ読み取りエラー: {e} ---\n"))
        finally:
            # 処理終了後にボタンの状態を更新
            if process == self.generate_process:
                self.after(0, self.stop_generation, False) # ユーザーによる停止ではない
            elif process == self.search_process:
                self.after(0, self.stop_search, False) # ユーザーによる停止ではない


    # --- Dummy methods for now ---
    def browse_directory(self):
        # フォルダ選択ダイアログを開く
        # 初期ディレクトリは現在の設定値、なければカレントディレクトリ
        initial_dir = self.target_path.get() if self.target_path.get() else os.getcwd()
        path = filedialog.askdirectory(
            title="処理対象のフォルダを選択",
            initialdir=initial_dir
            )
        if path:
            self.target_path.set(path)

    def start_generation(self):
        path = self.target_path.get()
        if not path or not os.path.isdir(path):
            messagebox.showerror("エラー", "有効な処理対象パスを指定してください。")
            return

        self.generate_button.configure(state="disabled")
        self.generate_stop_button.configure(state="normal")
        self.generate_output_text.delete("1.0", "end")
        self.generate_output_text.insert("end", f"--- 生成処理を開始します ---\n")
        self.generate_output_text.insert("end", f"対象パス: {path}\n")

        # コマンドを構築
        command = [
            "python", "-u", "local_image_super_search2.py", "generate"
        ]
        
        # 実行ファイルがある場合はそちらを優先
        executable_path = resource_path(EXECUTABLE_NAME)
        if os.path.exists(executable_path):
             command = [executable_path, "generate"]

        generate_count = self.generate_count_entry.get()
        if generate_count.isdigit():
            command.append(generate_count)
            self.generate_output_text.insert("end", f"生成件数: {generate_count}\n")
        
        self.generate_output_text.insert("end", f"実行コマンド: {' '.join(command)}\n--------------------\n")

        try:
            # プロセスを非同期で開始
            output_encoding = locale.getpreferredencoding(False) if platform.system() == "Windows" else "utf-8"
            self.generate_process = subprocess.Popen(
                command,
                cwd=path, # カレントディレクトリを処理対象パスに設定
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=output_encoding,
                errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            
            # 出力を別スレッドで読み取る
            threading.Thread(
                target=self._read_process_output,
                args=(self.generate_process, self.generate_output_text),
                daemon=True
            ).start()

        except FileNotFoundError:
            messagebox.showerror("エラー", f"実行ファイル '{command[0]}' が見つかりません。")
            self.stop_generation(user_stopped=False)
        except Exception as e:
            messagebox.showerror("エラー", f"処理の開始に失敗しました: {e}")
            self.stop_generation(user_stopped=False)


    def stop_generation(self, user_stopped=True):
        if self.generate_process and self.generate_process.poll() is None:
            try:
                self.generate_process.kill() # プロセスを強制終了
                self.generate_process.wait() # 終了を待つ
                if user_stopped:
                    self.generate_output_text.insert("end", "\n--- ユーザーによって処理が停止されました ---\n")
            except Exception as e:
                self.generate_output_text.insert("end", f"\n--- プロセスの停止に失敗: {e} ---\n")
        
        self.generate_process = None
        self.generate_button.configure(state="normal")
        self.generate_stop_button.configure(state="disabled")
        if not user_stopped:
            self.generate_output_text.insert("end", "\n--- 処理が完了しました ---\n")
        self.generate_output_text.see(tk.END)


    def start_search(self):
        query = self.query_entry.get()
        if not query:
            messagebox.showerror("エラー", "クエリ(検索語)を入力してください。")
            return

        # 処理対象パスも検索には必要（埋め込みファイルがそこにあるため）
        path = self.target_path.get()
        if not path or not os.path.isdir(path):
            messagebox.showerror("エラー", "有効な処理対象パスを指定してください。\n(検索対象の'local_image_embeddings.json'があるフォルダ)")
            return

        self.search_button.configure(state="disabled")
        self.search_stop_button.configure(state="normal")
        self.search_output_text.delete("1.0", "end")
        self.search_output_text.insert("end", f"--- 検索処理を開始します ---\n")
        self.search_output_text.insert("end", f"対象パス: {path}\n")

        # コマンドを構築
        command = [
            "python", "-u", "local_image_super_search2.py", "search", query
        ]
        
        # 実行ファイルがある場合はそちらを優先
        executable_path = resource_path(EXECUTABLE_NAME)
        if os.path.exists(executable_path):
             command = [executable_path, "search", query]

        # オプションを追加
        if self.no_word_var.get():
            command.append("-noword")
        if self.no_exword_var.get():
            command.append("-noexword")
        if self.no_cos_var.get():
            command.append("-nocos")
        
        n_value = self.n_entry.get()
        if n_value.isdigit():
            command.extend(["-n", n_value])
            
        t_value = self.t_entry.get()
        if t_value:
            command.extend(["-t", t_value])
            
        # 表示用のコマンド文字列を作成（クエリを引用符で囲む）
        log_command_display = command[:]
        # クエリは 'search' の次にある
        try:
            search_index = log_command_display.index('search')
            query_index = search_index + 1
            log_command_display[query_index] = f'"{log_command_display[query_index]}"'
        except ValueError:
            pass # 'search' が見つからない場合は何もしない（念のため）

        self.search_output_text.insert("end", f"実行コマンド: {' '.join(log_command_display)}\n--------------------\n")

        try:
            # プロセスを非同期で開始
            output_encoding = locale.getpreferredencoding(False) if platform.system() == "Windows" else "utf-8"
            self.search_process = subprocess.Popen(
                command,
                cwd=path, # カレントディレクトリを処理対象パスに設定
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=output_encoding,
                errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )
            
            # 出力を別スレッドで読み取る
            threading.Thread(
                target=self._read_process_output,
                args=(self.search_process, self.search_output_text),
                daemon=True
            ).start()

        except FileNotFoundError:
            messagebox.showerror("エラー", f"実行ファイル '{command[0]}' が見つかりません。")
            self.stop_search(user_stopped=False)
        except Exception as e:
            messagebox.showerror("エラー", f"処理の開始に失敗しました: {e}")
            self.stop_search(user_stopped=False)


    def stop_search(self, user_stopped=True):
        if self.search_process and self.search_process.poll() is None:
            try:
                self.search_process.kill()
                self.search_process.wait()
                if user_stopped:
                    self.search_output_text.insert("end", "\n--- ユーザーによって処理が停止されました ---\n")
            except Exception as e:
                 self.search_output_text.insert("end", f"\n--- プロセスの停止に失敗: {e} ---\n")

        self.search_process = None
        self.search_button.configure(state="normal")
        self.search_stop_button.configure(state="disabled")
        if not user_stopped:
            self.search_output_text.insert("end", "\n--- 処理が完了しました ---\n")
        self.search_output_text.see(tk.END)


    def load_readme(self):
        # Placeholder
        if not os.path.exists(README_FILE):
            # READMEがない場合、上位のディレクトリも探す
            if os.path.exists(os.path.join("..", README_FILE)):
                readme_path = os.path.join("..", README_FILE)
            else:
                 messagebox.showerror("エラー", f"{README_FILE} が見つかりません。")
                 return
        else:
            readme_path = README_FILE

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            self.readme_text.configure(state="normal")
            self.readme_text.delete("1.0", "end")
            self.readme_text.insert("1.0", readme_content)
            self.readme_text.configure(state="disabled")
        except FileNotFoundError:
            messagebox.showerror("エラー", f"{readme_path} が見つかりません。")

    def save_config(self):
        config = configparser.ConfigParser()
        config['DEFAULT'] = {'target_path': self.target_path.get()}
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)

    def load_config(self):
        config = configparser.ConfigParser()
        if os.path.exists(CONFIG_FILE):
            config.read(CONFIG_FILE)
            path = config['DEFAULT'].get('target_path', '')
            self.target_path.set(path)

    def on_closing(self):
        self.save_config()
        # Kill any running subprocesses
        if self.generate_process and self.generate_process.poll() is None:
            self.generate_process.kill()
        if self.search_process and self.search_process.poll() is None:
            self.search_process.kill()
        self.destroy()


if __name__ == "__main__":
    # Check if customtkinter is installed
    try:
        import customtkinter
    except ImportError:
        # A simple tkinter window to show the error
        root = tk.Tk()
        root.withdraw() # hide the root window
        messagebox.showerror("依存関係エラー", "CustomTkinterがインストールされていません。\n\npip install customtkinter\n\n上記のコマンドを実行してください。")
        exit()
        
    app = App()
    app.mainloop()
