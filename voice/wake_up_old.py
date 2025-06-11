#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden  
# @Date: 2024/11/27

import time
import threading
import math

class SimpleAudioWakeup:
    """
    简单音频唤醒检测器
    
    这个检测器不需要复杂的语音识别库，而是通过检测声音的模式来工作
    就像训练一只狗识别特定的敲门模式一样，我们检测特定的声音模式
    
    工作原理：
    1. 持续监听环境中的音量变化
    2. 当检测到符合"你好"音节模式的声音时触发唤醒
    3. "你好"通常是两个音节，每个音节约0.3-0.5秒，中间有短暂停顿
    """
    
    def __init__(self):
        # 尝试导入pyaudio，这是我们的"耳朵"
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
        
        # 音频参数配置 - 这些参数就像调节收音机的频道
        self.chunk_size = 1024      # 每次读取的音频数据块大小
        self.sample_rate = 44100    # 采样率，类似于照片的分辨率
        self.channels = 1           # 单声道，简化处理
        
        # 声音检测的关键参数
        self.base_threshold = 1000   # 基础音量阈值
        self.dynamic_threshold = 1000 # 动态调整的阈值
        self.silence_threshold = 500  # 静音阈值
        
        # 模式识别参数 - 用于识别"你好"的双音节模式
        self.min_syllable_duration = 0.2  # 每个音节最短持续时间（秒）
        self.max_syllable_duration = 0.8   # 每个音节最长持续时间（秒）
        self.max_gap_duration = 0.3        # 音节间最大间隔时间（秒）
        self.min_activation_count = 2       # 需要检测到的音节数量
        
        # 状态变量
        self.is_listening = False
        self.audio_stream = None
        self.background_noise_level = 0
        
        print(f"🎤 简单音频唤醒检测器已初始化")
        print(f"目标检测: 双音节模式（如'你好'）")
    
    def _calculate_volume(self, audio_data):
        """
        计算音频数据的音量（RMS值）
        
        这个函数就像是一个音量表，告诉我们声音有多大
        RMS（均方根）是测量音频强度的标准方法
        """
        if len(audio_data) == 0:
            return 0
        
        # 将字节数据转换为数值，然后计算均方根
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # 计算RMS值 - 这个数学公式能准确反映声音的"能量"
        rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
        return rms
    
    def _calibrate_background_noise(self):
        """
        校准背景噪音级别
        
        这个过程就像是教会程序"什么是安静"
        它会持续监听几秒钟，学习当前环境的基础噪音水平
        """
        print("\n🔧 正在校准背景噪音...")
        print("请保持安静3秒钟，让系统学习环境噪音...")
        
        noise_samples = []
        calibration_duration = 3.0  # 校准持续3秒
        samples_needed = int(calibration_duration * self.sample_rate / self.chunk_size)
        
        for i in range(samples_needed):
            try:
                # 读取一小段音频数据
                audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                volume = self._calculate_volume(audio_data)
                noise_samples.append(volume)
                
                # 显示进度，让用户知道程序在工作
                progress = (i + 1) / samples_needed * 100
                if i % 10 == 0:  # 每隔一段时间显示进度
                    print(f"校准进度: {progress:.1f}%")
                    
            except Exception as e:
                print(f"校准过程中出现错误: {e}")
                break
        
        if noise_samples:
            # 计算背景噪音的平均水平
            import numpy as np
            self.background_noise_level = np.mean(noise_samples)
            
            # 根据背景噪音动态设置检测阈值
            # 这就像是根据环境亮度调节相机的感光度
            self.dynamic_threshold = self.background_noise_level * 2.5
            
            print(f"✅ 背景噪音校准完成")
            print(f"背景噪音级别: {self.background_noise_level:.1f}")
            print(f"动态检测阈值: {self.dynamic_threshold:.1f}")
            
            # 给用户一些关于环境的建议
            if self.background_noise_level < 300:
                print("📢 环境很安静，检测效果应该很好")
            elif self.background_noise_level < 800:
                print("📢 环境有轻微噪音，但应该可以正常工作")
            else:
                print("📢 环境比较嘈杂，可能需要更大声说话")
                print("💡 建议: 尽量在安静环境中使用，或靠近麦克风")
        else:
            print("⚠️ 校准失败，将使用默认设置")
    
    def _detect_syllable_pattern(self, volume_history, time_history):
        """
        检测双音节模式（如"你好"）
        
        这个函数就像是一位音乐家，能够识别节拍和韵律
        它寻找符合中文双音节词汇特征的声音模式
        
        参数:
        volume_history: 最近的音量历史记录
        time_history: 对应的时间戳历史记录
        """
        if len(volume_history) < 10:  # 需要足够的数据来分析模式
            return False
        
        # 找出声音活动的时间段（音量超过阈值的时段）
        active_periods = []
        current_period_start = None
        
        for i, (volume, timestamp) in enumerate(zip(volume_history, time_history)):
            if volume > self.dynamic_threshold:
                # 声音开始
                if current_period_start is None:
                    current_period_start = timestamp
            else:
                # 声音结束
                if current_period_start is not None:
                    period_duration = timestamp - current_period_start
                    # 只记录合理长度的声音片段
                    if self.min_syllable_duration <= period_duration <= self.max_syllable_duration:
                        active_periods.append((current_period_start, timestamp, period_duration))
                    current_period_start = None
        
        # 检查是否有符合双音节模式的声音
        if len(active_periods) >= 2:
            # 检查最后两个音节的间隔是否合理
            last_two = active_periods[-2:]
            gap_duration = last_two[1][0] - last_two[0][1]  # 第二个音节开始 - 第一个音节结束
            
            if 0 < gap_duration <= self.max_gap_duration:
                print(f"🎯 检测到双音节模式！")
                print(f"   第一音节: {last_two[0][2]:.2f}秒")
                print(f"   间隔: {gap_duration:.2f}秒") 
                print(f"   第二音节: {last_two[1][2]:.2f}秒")
                return True
        
        return False
    
    def _on_wake_detected(self):
        """
        当检测到唤醒模式时的响应
        
        这里是检测成功后的处理逻辑，您可以根据需要自定义
        """
        print("\n" + "="*60)
        print("🔊 检测到'你好'音节模式！")
        print("📱 设备已唤醒，准备接收指令...")
        print("="*60)
        
        # 在这里添加您希望在唤醒后执行的操作
        # 比如：
        # - 播放确认音效
        # - 启动其他程序模块
        # - 发送通知
        # - 控制硬件设备等
        
        print("👋 你好！系统已激活...")
        
        # 短暂暂停，避免重复触发
        time.sleep(2)
    
    def start_listening(self):
        """
        开始监听音频模式
        
        这是程序的核心工作循环，像一位专注的守卫
        持续监听音频输入，寻找特定的声音模式
        """
        if not self.audio_available:
            print("❌ 音频系统不可用，无法开始监听")
            return
        
        print(f"\n🎧 开始监听双音节模式...")
        print("💡 使用说明:")
        print("   - 清晰地说'你好'或类似的双音节词")
        print("   - 每个音节要清晰分开")
        print("   - 尽量保持适中的音量")
        print("   - 按 Ctrl+C 退出")
        
        try:
            # 初始化音频流 - 这就像打开我们的"耳朵"
            self.audio_stream = self.pyaudio.PyAudio().open(
                format=self.pyaudio.paInt16,   # 16位音频格式
                channels=self.channels,         # 单声道
                rate=self.sample_rate,         # 采样率
                input=True,                    # 输入模式（录音）
                frames_per_buffer=self.chunk_size
            )
            
            # 校准背景噪音
            self._calibrate_background_noise()
            
            print(f"\n正在监听中... (检测阈值: {self.dynamic_threshold:.1f})")
            
            # 用于存储音量和时间历史的列表
            volume_history = []
            time_history = []
            history_limit = 100  # 只保留最近100个数据点
            
            self.is_listening = True
            check_count = 0
            
            while self.is_listening:
                try:
                    # 读取音频数据
                    audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    current_time = time.time()
                    
                    # 计算当前音量
                    volume = self._calculate_volume(audio_data)
                    
                    # 更新历史记录
                    volume_history.append(volume)
                    time_history.append(current_time)
                    
                    # 保持历史记录在合理范围内
                    if len(volume_history) > history_limit:
                        volume_history.pop(0)
                        time_history.pop(0)
                    
                    # 定期显示状态信息
                    check_count += 1
                    if check_count % 50 == 0:  # 每50次检测显示一次状态
                        print(f"⏳ 监听中... 当前音量: {volume:.1f} (阈值: {self.dynamic_threshold:.1f})")
                    
                    # 如果检测到足够的音量，开始分析模式
                    if volume > self.dynamic_threshold:
                        print(f"🔍 检测到声音活动: {volume:.1f}")
                        
                        # 检查是否符合双音节模式
                        if self._detect_syllable_pattern(volume_history, time_history):
                            self._on_wake_detected()
                            
                            # 清空历史记录，避免重复检测
                            volume_history.clear()
                            time_history.clear()
                
                except Exception as e:
                    print(f"⚠️ 音频处理出错: {e}")
                    time.sleep(0.1)
                    continue
                    
        except KeyboardInterrupt:
            print("\n👋 检测到退出信号...")
            
        except Exception as e:
            print(f"❌ 音频系统错误: {e}")
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
                print("🔇 音频流已关闭")
            except:
                pass
        
        print("⏹️ 监听已停止")
        print("感谢使用简单音频唤醒检测器！")


def main():
    """
    主程序入口
    这里是整个程序开始的地方
    """
    print("🎤 简单音频唤醒检测器")
    print("=" * 50)
    print("功能说明:")
    print("- 无需复杂的语音识别库")
    print("- 通过声音模式检测双音节词汇")
    print("- 适合检测'你好'等中文词汇")
    print("- 仅需要pyaudio库支持")
    print("=" * 50)
    
    # 首先检查必要的依赖
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
