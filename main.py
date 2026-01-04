from transformers import CLIPProcessor, CLIPModel
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import argparse
import shutil
from pathlib import Path
import PyPDF2
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PIL import Image
import uuid
import itertools
import torch


class SimpleAIAssistant:
    """AI助手 - 修复版"""

    def __init__(self):
        # 创建数据目录
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        print("正在初始化模型...")

        # 1. 文本模型
        try:
            self.text_model = SentenceTransformer('./models/paraphrase-albert-small-v2')
            print("文本模型加载成功")
        except Exception as e:
            print(f"文本模型加载失败: {e}")
            self.text_model = None

        # 2. CLIP模型
        self.clip_model = None
        self.clip_processor = None
        try:
            print("正在加载CLIP模型...")
            self.clip_model = CLIPModel.from_pretrained("./models/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("./models/clip-vit-base-patch32")
            print("CLIP模型加载成功")
        except Exception as e:
            print(f"CLIP模型加载失败: {e}")
            print("  图片搜索功能将受限")

        # 3. 向量数据库
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.data_dir / "vector_db"),
            settings=Settings(anonymized_telemetry=False)
        )

        # 创建集合
        self.paper_collection = self.chroma_client.get_or_create_collection("papers")
        self.image_collection = self.chroma_client.get_or_create_collection("images")

        print("系统初始化完成\n")

    def extract_pdf_text(self, pdf_path, max_pages=5):
        """从PDF提取文本"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf = PyPDF2.PdfReader(file)
                for i in range(min(len(pdf.pages), max_pages)):
                    text += pdf.pages[i].extract_text() + "\n"
        except Exception as e:
            print(f"读取PDF失败: {e}")
        return text

    def _simple_classify(self, text):
        """简单分类器（基于关键词）"""
        text_lower = text.lower()
        topics = []

        # 定义关键词
        keywords = {
            "CV": ["vision", "image", "detection", "segmentation"],
            "NLP": ["language", "text", "transformer", "attention"],
            "RL": ["reinforcement", "reward", "agent", "policy"],
            "ML": ["learning", "neural", "network", "model"]
        }

        for topic, words in keywords.items():
            if any(word in text_lower for word in words):
                topics.append(topic)

        return topics if topics else ["其他"]

    def add_paper(self, pdf_path):
        """添加一篇论文"""
        if not pdf_path.endswith('.pdf'):
            print("请提供PDF文件")
            return

        print(f"处理: {pdf_path}")

        # 1. 提取文本
        text = self.extract_pdf_text(pdf_path)
        if not text:
            print("无法提取文本")
            return

        # 2. 生成向量
        vector = self.text_model.encode(text[:500]).tolist()  # 只取前500字符

        # 3. 简单分类（基于关键词）
        topics = self._simple_classify(text)

        # 4. 存储到数据库
        paper_id = str(uuid.uuid4())
        self.paper_collection.add(
            ids=[paper_id],
            embeddings=[vector],
            metadatas=[{
                "file": os.path.basename(pdf_path),
                "path": pdf_path,
                "topics": ",".join(topics)
            }]
        )

        # 5. 复制到分类文件夹
        for topic in topics:
            topic_dir = self.data_dir / "papers" / topic
            topic_dir.mkdir(parents=True, exist_ok=True)
            dest_path = topic_dir / os.path.basename(pdf_path)
            if not dest_path.exists():  # 避免重复复制
                shutil.copy2(pdf_path, dest_path)

        print(f"已添加: {os.path.basename(pdf_path)} -> {topics}")
        return paper_id

    def search_papers(self, query, n_results=3):
        """搜索论文 - 修复版，不显示相似度"""
        # 生成查询向量
        query_vector = self.text_model.encode(query).tolist()

        # 在数据库中搜索
        results = self.paper_collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )

        # 显示结果
        print(f"\n搜索: '{query}'")
        print("=" * 50)

        if results['ids'][0]:
            for i, (paper_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
                print(f"\n{i + 1}. {metadata['file']}")
                print(f"   主题: {metadata['topics']}")
                print(f"   路径: {metadata['path']}")
                # 删除相似度显示
        else:
            print("未找到相关论文")

        return results

    def add_image(self, image_path):
        """添加图片（使用文本描述）"""
        from transformers import CLIPProcessor, CLIPModel
        import torch

        # 检查CLIP模型是否加载
        if self.clip_model is None:
            print("CLIP模型未加载，无法添加图片")
            return None

        # 处理图片
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        # 存储到数据库
        image_id = str(uuid.uuid4())
        self.image_collection.add(
            ids=[image_id],
            embeddings=[image_features[0].tolist()],
            metadatas=[{
                "file": os.path.basename(image_path),
                "path": image_path
            }]
        )

        print(f"已添加图片: {os.path.basename(image_path)}")
        return image_id

    def search_images(self, query, n_results=3):
        """以文搜图 - 修复版，不显示相似度"""
        # 检查CLIP模型
        if self.clip_model is None:
            print("CLIP模型未加载，无法搜索图片")
            return

        print(f"\n图片搜索: '{query}'")
        print("=" * 50)

        try:
            # 用CLIP处理文本查询
            text_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                query_vector = text_features[0].tolist()

            # 搜索图片
            results = self.image_collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )

            if results['ids'][0]:
                print(f"找到 {len(results['ids'][0])} 张相关图片:")
                for i, (img_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
                    print(f"\n{i + 1}. {metadata['file']}")
                    print(f"   路径: {metadata['path']}")
                    # 删除相似度显示
            else:
                print("未找到相关图片")

        except Exception as e:
            print(f"搜索出错: {e}")

        return results

    def batch_add_papers(self, folder_path):
        """批量添加论文"""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))

        print(f"找到 {len(pdf_files)} 个PDF文件")

        for pdf_file in pdf_files:
            self.add_paper(str(pdf_file))

        print("\n批量处理完成！")

    def batch_add_images(self, folder_path):
        """批量添加图片 - 修复重复问题"""
        if self.clip_model is None:
            print("CLIP模型未加载，无法添加图片")
            return

        folder = Path(folder_path)

        # 查找所有图片文件，避免重复
        image_files = []
        seen_files = set()

        for ext in ['.jpg', '.jpeg', '.png']:
            for img_file in folder.glob(f"*{ext}"):
                if img_file.name not in seen_files:
                    image_files.append(img_file)
                    seen_files.add(img_file.name)

        print(f"找到 {len(image_files)} 张图片")

        for img_file in image_files:
            self.add_image(str(img_file))

        print("\n图片添加完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化版AI文献助手")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 添加论文命令
    add_parser = subparsers.add_parser('add', help='添加论文或图片')
    add_parser.add_argument('path', help='文件路径或文件夹路径')
    add_parser.add_argument('--type', choices=['paper', 'image'], default='paper',
                            help='文件类型：paper或image')

    # 搜索命令
    search_parser = subparsers.add_parser('search', help='搜索内容')
    search_parser.add_argument('query', help='搜索关键词')
    search_parser.add_argument('--type', choices=['paper', 'image'], default='paper',
                               help='搜索类型：paper或image')
    search_parser.add_argument('--n', type=int, default=3, help='返回结果数量')

    args = parser.parse_args()

    # 初始化助手
    assistant = SimpleAIAssistant()

    if args.command == 'add':
        path = Path(args.path)

        if args.type == 'paper':
            if path.is_dir():
                assistant.batch_add_papers(str(path))
            else:
                assistant.add_paper(str(path))
        else:  # image
            if path.is_dir():
                assistant.batch_add_images(str(path))
            else:
                assistant.add_image(str(path))

    elif args.command == 'search':
        if args.type == 'paper':
            assistant.search_papers(args.query, args.n)
        else:
            assistant.search_images(args.query, args.n)

    else:
        parser.print_help()
        print("\n使用示例:")
        print("  添加论文: python main.py add paper.pdf")
        print("  批量添加: python main.py add papers_folder/")
        print("  搜索论文: python main.py search '机器学习'")
        print("  搜索图片: python main.py search '猫' --type image")


if __name__ == "__main__":
    main()