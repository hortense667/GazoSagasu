import os
import json
from datetime import datetime
from PIL import Image, ExifTags
import re

def get_image_creation_date(image_path):
    """画像の作成日時を取得（EXIF > ファイル名 > ファイル更新日時）"""
    # 1. EXIFから撮影日時を取得
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'DateTimeOriginal':
                    return datetime.strptime(value, '%Y:%m:%d %H:%M:%S').isoformat()
    except Exception:
        pass

    # 2. ファイル名から日付らしき文字列を抽出
    filename = os.path.basename(image_path)
    match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
    if match:
        try:
            year, month, day = map(int, match.groups())
            return datetime(year, month, day).isoformat()
        except ValueError:
            pass

    # 3. ファイルの最終更新日時を使用
    try:
        mtime = os.path.getmtime(image_path)
        return datetime.fromtimestamp(mtime).isoformat()
    except Exception:
        return None

def migrate_embeddings(embeddings_file="local_image_embeddings.json"):
    """既存の埋め込みデータにimage_creation_dateを追加する"""
    if not os.path.exists(embeddings_file):
        print(f"エラー: {embeddings_file} が見つかりません。")
        return

    try:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"エラー: {embeddings_file} が破損しています: {e}")
        return

    updated_count = 0
    missing_files = 0

    for image_file, data in embeddings_data.items():
        if 'image_creation_date' not in data or data['image_creation_date'] is None:
            if os.path.exists(image_file):
                print(f"更新中: {image_file}")
                creation_date = get_image_creation_date(image_file)
                data['image_creation_date'] = creation_date
                updated_count += 1
            else:
                print(f"警告: ファイルが見つかりません: {image_file}")
                missing_files += 1

    if updated_count > 0:
        try:
            # バックアップを作成
            backup_file = embeddings_file + ".bak"
            os.rename(embeddings_file, backup_file)
            print(f"元のファイルを {backup_file} にバックアップしました。")

            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
            print(f"\n完了！ {updated_count} 件のデータに作成日を追加しました。")
        except Exception as e:
            print(f"\nエラー: ファイルの保存中にエラーが発生しました: {e}")
            # エラーが発生した場合はバックアップを復元
            if os.path.exists(backup_file):
                os.rename(backup_file, embeddings_file)
                print("バックアップからファイルを復元しました。")
    else:
        print("すべてのデータに作成日が含まれているため、更新は不要です。")

    if missing_files > 0:
        print(f"\n警告: {missing_files} 個の画像ファイルが見つかりませんでした。")


if __name__ == "__main__":
    print("=" * 50)
    print("埋め込みデータ移行ツール")
    print("=" * 50)
    print("local_image_embeddings.json に画像の作成日を追加します。")
    
    migrate_embeddings()