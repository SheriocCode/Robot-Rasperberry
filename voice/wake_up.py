import time
import threading
import math
import numpy as np
import os
import sys

class SimpleAudioWakeup:
    """
    优化版简单音频唤醒检测器
    
    这个版本通过实时模式检测和可视化反馈，提高了检测速度和用户体验
    """
    
    def __init__(self):
        # 尝试导入pyaudio
        try:
            import pyaudio
            self.pyaudio = pyaudio
            self.audio_available = True
            print("✅ 音频系统初始化成功")
        except ImportError:
            print("❌ 无法导入pyaudio库")
            print("请运行: pip install pyaudio")
            self.audio_available = False
            return
        
        # 音频参数配置
        self.chunk_size = 1024      
        self.sample_rate = 44100    
        self.channels = 1           
        self.format = self.pyaudio.paInt16
        
        # 声音检测的关键参数 - 调整为更敏感的值
        self.base_threshold = 800   # 降低基础阈值以提高灵敏度
        self.dynamic_threshold = 800
        self.silence_threshold = 400
        
        # 模式识别参数 - 优化为更适合中文双音节
        self.min_syllable_duration = 0.15  # 缩短最小音节时长
        self.max_syllable_duration = 0.7   # 调整最大音节时长
        self.max_gap_duration = 0.25       # 缩短最大间隔时间
        self.min_activation_count = 2      # 仍需检测到两个音节
        
        # 可视化参数
        self.volume_bar_length = 50
        self.volume_max_display = 5000     # 音量显示的最大值
        self.visualization_update_rate = 0.05  # 可视化更新频率(秒)
        
        # 状态变量
        self.is_listening = False
        self.audio_stream = None
        self.background_noise_level = 0
        self.last_volume = 0
        
        # 音节检测状态
        self.in_syllable = False
        self.syllable_start_time = 0
        self.syllable_end_time = 0
        self.syllables_detected = []
        
        print(f"🎤 优化版简单音频唤醒检测器已初始化")
        print(f"目标检测: 双音节模式（如'你好'）")
    
    def _calculate_volume(self, audio_data):
        """计算音频数据的音量（RMS值）"""
        if len(audio_data) == 0:
            return 0
        
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
        return rms
    
    def _calibrate_background_noise(self):
        """校准背景噪音级别"""
        print("\n🔧 正在校准背景噪音...")
        print("请保持安静3秒钟，让系统学习环境噪音...")
        
        noise_samples = []
        calibration_duration = 3.0
        samples_needed = int(calibration_duration * self.sample_rate / self.chunk_size)
        
        # 显示进度条
        self._show_progress_bar(0, samples_needed, prefix="校准进度:")
        
        for i in range(samples_needed):
            try:
                audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                volume = self._calculate_volume(audio_data)
                noise_samples.append(volume)
                
                # 更新进度条
                if i % 5 == 0:
                    self._show_progress_bar(i + 1, samples_needed, prefix="校准进度:")
                    
            except Exception as e:
                print(f"校准过程中出现错误: {e}")
                break
        
        if noise_samples:
            self.background_noise_level = np.mean(noise_samples)
            self.dynamic_threshold = max(self.background_noise_level * 2.5, self.base_threshold)
            
            print(f"\n✅ 背景噪音校准完成")
            print(f"背景噪音级别: {self.background_noise_level:.1f}")
            print(f"动态检测阈值: {self.dynamic_threshold:.1f}")
            
            # 给用户环境建议
            if self.background_noise_level < 300:
                print("📢 环境很安静，检测效果应该很好")
            elif self.background_noise_level < 800:
                print("📢 环境有轻微噪音，但应该可以正常工作")
            else:
                print("📢 环境比较嘈杂，可能需要更大声说话")
                print("💡 建议: 尽量在安静环境中使用，或靠近麦克风")
        else:
            print("⚠️ 校准失败，将使用默认设置")
    
    def _show_progress_bar(self, iteration, total, prefix='', suffix='', length=50, fill='█'):
        """显示进度条"""
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
        # 当完成时添加换行
        if iteration == total: 
            print()
    
    def _detect_syllable_pattern(self):
        """实时检测双音节模式"""
        # 确保有足够的音节
        if len(self.syllables_detected) < 2:
            return False
        
        # 获取最后两个音节
        last_two = self.syllables_detected[-2:]
        
        # 检查间隔时间
        gap_duration = last_two[1][0] - last_two[0][1]  # 第二个音节开始 - 第一个音节结束
        
        if 0 < gap_duration <= self.max_gap_duration:
            # 打印检测信息
            self._clear_line()
            print(f"🎯 检测到双音节模式！")
            print(f"   第一音节: {last_two[0][2]:.2f}秒")
            print(f"   间隔: {gap_duration:.2f}秒") 
            print(f"   第二音节: {last_two[1][2]:.2f}秒")
            return True
        
        return False
    
    def _on_wake_detected(self):
        """当检测到唤醒模式时的响应"""
        self._clear_screen()
        print("\n" + "="*60)
        print("🔊 检测到'你好'音节模式！")
        print("📱 设备已唤醒，准备接收指令...")
        print("="*60)
        
        print("👋 你好！系统已激活...")
        
        # 短暂暂停，避免重复触发
        time.sleep(2)
        
        # 重置检测状态
        self.syllables_detected = []
    
    def _clear_line(self):
        """清除当前行"""
        sys.stdout.write("\033[K")  # 清除到行尾
        sys.stdout.flush()
    
    def _clear_screen(self):
        """清除屏幕"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _display_volume_visualization(self, volume):
        """显示音量可视化"""
        # 计算音量条长度
        bar_length = int(min(1.0, volume / self.volume_max_display) * self.volume_bar_length)
        
        # 确定音量条颜色/符号
        if volume < self.silence_threshold:
            bar_char = ' '
            status = "安静"
        elif volume < self.dynamic_threshold:
            bar_char = '░'
            status = "背景音"
        else:
            bar_char = '█'
            status = "检测中"
        
        # 创建音量条
        volume_bar = bar_char * bar_length + ' ' * (self.volume_bar_length - bar_length)
        
        # 显示状态信息
        status_text = f"音量: {volume:5.1f} | 阈值: {self.dynamic_threshold:5.1f} | 状态: {status}"
        
        # 打印音量条和状态
        self._clear_line()
        print(f"[{volume_bar}] {status_text}", end='\r')
        sys.stdout.flush()
    
    def start_listening(self):
        """开始监听音频模式"""
        if not self.audio_available:
            print("❌ 音频系统不可用，无法开始监听")
            return
        
        self._clear_screen()
        print(f"🎧 开始监听双音节模式...")
        print("💡 使用说明:")
        print("   - 清晰地说'你好'或类似的双音节词")
        print("   - 每个音节要清晰分开")
        print("   - 尽量保持适中的音量")
        print("   - 按 Ctrl+C 退出")
        
        try:
            # 初始化音频流
            self.audio_stream = self.pyaudio.PyAudio().open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # 校准背景噪音
            self._calibrate_background_noise()
            
            print(f"\n正在监听中...")
            
            self.is_listening = True
            last_visualization_time = time.time()
            
            while self.is_listening:
                try:
                    # 读取音频数据
                    audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    current_time = time.time()
                    
                    # 计算当前音量
                    volume = self._calculate_volume(audio_data)
                    self.last_volume = volume
                    
                    # 更新可视化（限制更新频率）
                    if current_time - last_visualization_time > self.visualization_update_rate:
                        self._display_volume_visualization(volume)
                        last_visualization_time = current_time
                    
                    # 音节检测逻辑
                    if volume > self.dynamic_threshold:
                        # 开始或继续音节
                        if not self.in_syllable:
                            self.in_syllable = True
                            self.syllable_start_time = current_time
                    else:
                        # 结束音节
                        if self.in_syllable:
                            self.in_syllable = False
                            self.syllable_end_time = current_time
                            syllable_duration = self.syllable_end_time - self.syllable_start_time
                            
                            # 只记录合理长度的音节
                            if (self.min_syllable_duration <= syllable_duration <= 
                                self.max_syllable_duration):
                                self.syllables_detected.append(
                                    (self.syllable_start_time, self.syllable_end_time, syllable_duration)
                                )
                                
                                # 限制历史记录长度
                                if len(self.syllables_detected) > 10:
                                    self.syllables_detected.pop(0)
                            
                            # 检测是否符合双音节模式
                            if self._detect_syllable_pattern():
                                self._on_wake_detected()
                
                except Exception as e:
                    print(f"\n⚠️ 音频处理出错: {e}")
                    time.sleep(0.1)
                    continue
                    
        except KeyboardInterrupt:
            print("\n👋 检测到退出信号...")
            
        except Exception as e:
            print(f"\n❌ 音频系统错误: {e}")
            print("💡 建议检查:")
            print("   - 麦克风是否正常连接")
            print("   - 系统音频权限设置")
            print("   - pyaudio库是否正确安装")
            
        finally:
            self.stop_listening()
    
    def stop_listening(self):
        """安全停止监听"""
        self.is_listening = False
        
        if hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                print("\n🔇 音频流已关闭")
            except:
                pass
        
        print("⏹️ 监听已停止")
        print("感谢使用优化版简单音频唤醒检测器！")


def main():
    """主程序入口"""
    print("🎤 优化版简单音频唤醒检测器")
    print("=" * 50)
    print("功能亮点:")
    print("- 实时模式检测，响应更快")
    print("- 音量可视化反馈")
    print("- 更直观的状态显示")
    print("- 通过声音模式检测双音节词汇")
    print("- 适合检测'你好'等中文词汇")
    print("=" * 50)
    
    # 检查必要的依赖
    try:
        import numpy
        print("✅ numpy库检查通过")
    except ImportError:
        print("❌ 缺少numpy库，请运行: pip install numpy")
        return
    
    try:
        import pyaudio
        print("✅ pyaudio库检查通过")
    except ImportError:
        print("❌ 缺少pyaudio库，请运行: pip install pyaudio")
        print("💡 如果安装遇到问题，可能需要安装系统级音频库")
        return
    
    print("✅ 依赖检查完成\n")
    
    try:
        # 创建并启动检测器
        detector = SimpleAudioWakeup()
        
        if detector.audio_available:
            detector.start_listening()
        else:
            print("无法启动音频检测器")
            
    except Exception as e:
        print(f"程序运行出错: {e}")
        
    finally:
        print("程序已结束")


if __name__ == "__main__":
    main()
