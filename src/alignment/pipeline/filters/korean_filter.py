import re
import numpy as np
from collections import Counter

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import split_into_words
from datatrove.utils.typeshelper import Languages


class KoreanQualityFilter(BaseFilter):
    name = "🇰🇷 Korean Quality"

    def __init__(
        self,
        min_doc_words: int | None = 30,                    # 최소 단어 수 (한국어는 교착어라 단어가 적을 수 있음)
        max_doc_words: int | None = 50000,                 # 최대 단어 수
        min_avg_char_length: int | None = 1.5,             # 최소 평균 문자 길이 (한글 특성상 짧음)
        max_avg_char_length: int | None = 8,               # 최대 평균 문자 길이
        max_foreign_char_ratio: float | None = 0.3,        # 최대 외국 문자 비율
        max_number_ratio: float | None = 0.3,              # 최대 숫자 비율
        max_special_char_ratio: float | None = 0.15,       # 최대 특수문자 비율
        max_repeated_char_lines_ratio: float | None = 0.3, # 최대 반복 문자 라인 비율 (ㅋㅋㅋ, ㅠㅠㅠ 등)
        max_bullet_lines_ratio: float | None = 0.8,        # 최대 불릿 라인 비율
        max_ellipsis_lines_ratio: float | None = 0.4,      # 최대 생략부호 라인 비율
        min_korean_particles: int | None = 1,              # 최소 한국어 조사 개수
        max_emoticon_ratio: float | None = 0.1,            # 최대 이모티콘 비율
        exclusion_writer: DiskWriter = None,
    ):
        """
        Korean-specific quality filter that considers unique characteristics of Korean text.
        
        Args:
            min_doc_words: Minimum number of words (Korean words can be longer due to agglutination)
            max_doc_words: Maximum number of words
            min_avg_char_length: Minimum average character length per word
            max_avg_char_length: Maximum average character length per word
            max_foreign_char_ratio: Maximum ratio of non-Korean characters
            max_number_ratio: Maximum ratio of numeric characters
            max_special_char_ratio: Maximum ratio of special characters
            max_repeated_char_ratio: Maximum ratio of repeated Korean characters (ㅋㅋㅋ, ㅠㅠㅠ)
            max_bullet_lines_ratio: Maximum ratio of lines starting with bullets
            max_ellipsis_lines_ratio: Maximum ratio of lines ending with ellipsis
            min_korean_particles: Minimum number of Korean particles (은/는, 이/가, 을/를 etc.)
            max_emoticon_ratio: Maximum ratio of emoticons and emoji
            exclusion_writer: Writer for excluded documents
        """
        super().__init__(exclusion_writer)
        
        self.min_doc_words = min_doc_words
        self.max_doc_words = max_doc_words
        self.min_avg_char_length = min_avg_char_length
        self.max_avg_char_length = max_avg_char_length
        self.max_foreign_char_ratio = max_foreign_char_ratio
        self.max_number_ratio = max_number_ratio
        self.max_special_char_ratio = max_special_char_ratio
        self.max_repeated_char_lines_ratio = max_repeated_char_lines_ratio
        self.max_bullet_lines_ratio = max_bullet_lines_ratio
        self.max_ellipsis_lines_ratio = max_ellipsis_lines_ratio
        self.min_korean_particles = min_korean_particles
        self.max_emoticon_ratio = max_emoticon_ratio
        
        # Korean character ranges
        self.korean_range = re.compile(r'[가-힣ㄱ-ㅎㅏ-ㅣ]')
        self.hanja_range = re.compile(r'[一-龯]')
        
        # Common Korean particles and endings
        self.korean_particles = {
            '은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과', '하고',
            '의', '도', '만', '까지', '부터', '보다', '처럼', '같이', '마다', '조차', '라도',
            '한테', '에게', '께', '로부터', '으로부터', '라서', '이라서', '니까', '기 때문에'
        }
        
        # Korean repeated characters pattern (ㅋㅋㅋ, ㅠㅠㅠ, ㅎㅎㅎ, etc.)
        self.repeated_korean_chars = re.compile(r'([ㅋㅎㅠㅜㅡㅗㅓㅏㅣ])\1{2,}')
        
        # Bullet point patterns (Korean often uses • ○ ◦ ▪ ▫ - *)
        self.bullet_pattern = re.compile(r'^\s*[•○◦▪▫\-\*]\s')
        
        # Korean ellipsis patterns
        self.ellipsis_pattern = re.compile(r'[\.]{3,}|…|[ㅠㅜ]{3,}')
        
    def _count_korean_chars(self, text: str) -> int:
        """Count Korean characters (Hangul + Hanja)"""
        korean_chars = len(self.korean_range.findall(text))
        hanja_chars = len(self.hanja_range.findall(text))
        return korean_chars + hanja_chars
    
    def _count_foreign_chars(self, text: str) -> int:
        """Count non-Korean characters (excluding numbers and basic punctuation)"""
        total_chars = len(text)
        korean_chars = self._count_korean_chars(text)
        numbers = len(re.findall(r'[0-9]', text))
        basic_punct = len(re.findall(r'[\s\.\,\!\?\;\:\(\)\[\]\{\}\"\'\-]', text))
        return max(0, total_chars - korean_chars - numbers - basic_punct)
    
    def _count_emoticons(self, text: str) -> int:
        # Count emoji (basic range)
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
        return emoji_count

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """
        Filter Korean documents based on quality heuristics
        
        Args:
            doc: Document to filter
            
        Returns:
            True if document passes all filters, False with reason if it fails
        """
        text = doc.text
        words = split_into_words(text, Languages.korean)
        n_words = len(words)
        
        if n_words == 0:
            return False, "korean_empty_doc"
        
        # Filter out words that are only punctuation
        content_words = [w for w in words if self.korean_range.search(w) or self.hanja_range.search(w) or re.search(r'[a-zA-Z0-9]', w)]
        n_content_words = len(content_words)
        
        # 1. Document length checks
        if self.min_doc_words and n_content_words < self.min_doc_words:
            return False, "korean_short_doc"
        if self.max_doc_words and n_content_words > self.max_doc_words:
            return False, "korean_long_doc"
        
        # 2. Average character length per word
        if content_words:
            avg_char_length = np.mean([len(w) for w in content_words])
            if self.min_avg_char_length and avg_char_length < self.min_avg_char_length:
                return False, "korean_below_avg_char_threshold"
            if self.max_avg_char_length and avg_char_length > self.max_avg_char_length:
                return False, "korean_above_avg_char_threshold"
        
        text_length = len(text)
        if text_length == 0:
            return False, "korean_empty_text"
        
        # 3. Foreign character ratio
        if self.max_foreign_char_ratio:
            foreign_chars = self._count_foreign_chars(text)
            if foreign_chars / text_length > self.max_foreign_char_ratio:
                return False, "korean_too_many_foreign_chars"
        
        # 4. Number ratio
        if self.max_number_ratio:
            numbers = len(re.findall(r'[0-9]', text))
            if numbers / text_length > self.max_number_ratio:
                return False, "korean_too_many_numbers"
        
        # 5. Special character ratio
        if self.max_special_char_ratio:
            special_chars = len(re.findall(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ一-龯\.\,\!\?\;\:\(\)\[\]\{\}\"\'\-]', text))
            if special_chars / text_length > self.max_special_char_ratio:
                return False, "korean_too_many_special_chars"
        
        # 6. Line-based checks
        lines = text.splitlines()
        if lines:
            # Repeated Korean characters lines
            if self.max_repeated_char_lines_ratio:
                repeated_lines = sum(1 for line in lines if self.repeated_korean_chars.search(line))
                if repeated_lines / len(lines) > self.max_repeated_char_lines_ratio:
                    return False, "korean_too_many_repeated_char_lines"
            
            # Bullet point lines
            if self.max_bullet_lines_ratio:
                bullet_lines = sum(1 for line in lines if self.bullet_pattern.match(line))
                if bullet_lines / len(lines) > self.max_bullet_lines_ratio:
                    return False, "korean_too_many_bullets"
            
            # Ellipsis ending lines
            if self.max_ellipsis_lines_ratio:
                ellipsis_lines = sum(1 for line in lines if self.ellipsis_pattern.search(line.rstrip()))
                if ellipsis_lines / len(lines) > self.max_ellipsis_lines_ratio:
                    return False, "korean_too_many_ellipsis"
        
        # 7. Korean particles check (to ensure it's actually Korean text)
        if self.min_korean_particles and len(self.korean_particles.intersection(set(words))) < self.min_korean_particles:
            return False, "korean_insufficient_particles"
            
        # 8. Emoticon ratio
        if self.max_emoticon_ratio:
            emoticon_count = self._count_emoticons(text)
            if emoticon_count / text_length > self.max_emoticon_ratio:
                return False, "korean_too_many_emoticons"
        
        return True