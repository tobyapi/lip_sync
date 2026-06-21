using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace TobyApi.LipSync
{
    public enum LipSyncVowel { A = 0, I = 1, U = 2, E = 3, O = 4 }
    public enum LipSyncClass { Rest = 0, Closed = 1, A = 2, I = 3, U = 4, E = 5, O = 6, Fricative = 7, Other = 8 }
    public enum LipSyncCueKind { TtsViseme = 1, LyricTiming = 2 }
    public enum LipSyncMapperKind : uint { Generic = 0, Vrm = 1, Arkit = 2, MetaHuman = 3 }

    [Flags]
    public enum LipSyncOptionsFlags : uint
    {
        None = 0,
        SingingMode = 1 << 0,
        TinyNn = 1 << 1,
        TimedCues = 1 << 2,
        RobustLoudness = 1 << 3,
        Gmm = 1 << 4
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LipSyncOptions
    {
        public uint sampleRate;
        public LipSyncOptionsFlags flags;
        public float metadataWeight;
        public float smoothing;
        public float loudnessAdaptation;

        public static LipSyncOptions Create(int sampleRate, bool singingMode, bool tinyNn, bool timedCues, bool robustLoudness, bool gmm = false)
        {
            LipSyncOptionsFlags flags = LipSyncOptionsFlags.None;
            if (singingMode) flags |= LipSyncOptionsFlags.SingingMode;
            if (tinyNn) flags |= LipSyncOptionsFlags.TinyNn;
            if (timedCues) flags |= LipSyncOptionsFlags.TimedCues;
            if (robustLoudness) flags |= LipSyncOptionsFlags.RobustLoudness;
            if (gmm) flags |= LipSyncOptionsFlags.Gmm;

            return new LipSyncOptions
            {
                sampleRate = (uint)Mathf.Max(1, sampleRate),
                flags = flags,
                metadataWeight = 0.55f,
                smoothing = singingMode ? 0.65f : 0.18f,
                loudnessAdaptation = 0.07f
            };
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LipSyncTimedCue
    {
        public float startSeconds;
        public float endSeconds;
        public uint classIndex;
        public float weight;
        public uint kind;

        public LipSyncTimedCue(float startSeconds, float endSeconds, LipSyncClass lipSyncClass, float weight, LipSyncCueKind kind)
        {
            this.startSeconds = startSeconds;
            this.endSeconds = endSeconds;
            classIndex = (uint)lipSyncClass;
            this.weight = Mathf.Clamp01(weight);
            this.kind = (uint)kind;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LipSyncFrame
    {
        public float rest;
        public float closed;
        public float a;
        public float i;
        public float u;
        public float e;
        public float o;
        public float fricative;
        public float other;
        public float jawOpen;
        public float vowelConfidence;
        public float f1Hz;
        public float f2Hz;

        public float GetPosterior(LipSyncClass lipSyncClass)
        {
            switch (lipSyncClass)
            {
                case LipSyncClass.Rest: return rest;
                case LipSyncClass.Closed: return closed;
                case LipSyncClass.A: return a;
                case LipSyncClass.I: return i;
                case LipSyncClass.U: return u;
                case LipSyncClass.E: return e;
                case LipSyncClass.O: return o;
                case LipSyncClass.Fricative: return fricative;
                case LipSyncClass.Other: return other;
                default: return 0f;
            }
        }

        public LipSyncClass BestClass
        {
            get
            {
                LipSyncFrame frame = this;
                LipSyncClass bestClass;
                return LipSync.TryGetBestClass(ref frame, out bestClass) ? bestClass : BestClassFallback;
            }
        }

        public bool TryGetLegacyVowel(out LipSyncVowel vowel)
        {
            LipSyncFrame frame = this;
            if (LipSync.TryGetLegacyVowel(ref frame, out vowel)) return true;
            return TryMapLegacyVowelFallback(BestClassFallback, out vowel);
        }

        private LipSyncClass BestClassFallback
        {
            get
            {
                LipSyncClass bestClass = LipSyncClass.Rest;
                float bestScore = rest;
                for (int index = 1; index <= (int)LipSyncClass.Other; index++)
                {
                    LipSyncClass candidate = (LipSyncClass)index;
                    float score = GetPosterior(candidate);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = candidate;
                    }
                }
                return bestClass;
            }
        }

        private static bool TryMapLegacyVowelFallback(LipSyncClass lipSyncClass, out LipSyncVowel vowel)
        {
            switch (lipSyncClass)
            {
                case LipSyncClass.A: vowel = LipSyncVowel.A; return true;
                case LipSyncClass.I: vowel = LipSyncVowel.I; return true;
                case LipSyncClass.U: vowel = LipSyncVowel.U; return true;
                case LipSyncClass.E: vowel = LipSyncVowel.E; return true;
                case LipSyncClass.O: vowel = LipSyncVowel.O; return true;
                default: vowel = default(LipSyncVowel); return false;
            }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LipSyncMappedFrame
    {
        public uint kind;
        public uint bestClass;
        public int legacyVowel;
        public float confidence;
        public float jawOpen;
        public float aa;
        public float ih;
        public float ou;
        public float ee;
        public float oh;
        public float mouthClose;
        public float mouthFunnel;
        public float mouthPucker;
        public float mouthWide;
        public float mouthSmileLeft;
        public float mouthSmileRight;
        public float mouthLowerDownLeft;
        public float mouthLowerDownRight;
        public float mouthUpperUpLeft;
        public float mouthUpperUpRight;
        public float mouthPressLeft;
        public float mouthPressRight;
        public float fricative;

        public LipSyncMapperKind Kind => (LipSyncMapperKind)kind;
        public LipSyncClass BestClass => (LipSyncClass)bestClass;
        public bool HasLegacyVowel => legacyVowel >= 0;

        public static LipSyncMappedFrame Empty
        {
            get
            {
                LipSyncMappedFrame frame = default(LipSyncMappedFrame);
                frame.bestClass = (uint)LipSyncClass.Rest;
                frame.legacyVowel = -1;
                frame.confidence = 1f;
                return frame;
            }
        }

        public bool TryGetLegacyVowel(out LipSyncVowel vowel)
        {
            if (legacyVowel >= 0 && legacyVowel <= (int)LipSyncVowel.O)
            {
                vowel = (LipSyncVowel)legacyVowel;
                return true;
            }

            vowel = default(LipSyncVowel);
            return false;
        }
    }

    public sealed class LipSyncAnalyzer : IDisposable
    {
        private IntPtr handle;
        internal LipSyncAnalyzer(IntPtr handle) { this.handle = handle; }
        public bool IsValid => handle != IntPtr.Zero;
        public bool Process(float[] pcm, out LipSyncFrame frame) { return LipSync.TryProcessAnalyzer(handle, pcm, out frame); }
        public bool ProcessAtTime(float[] pcm, float timeSeconds, out LipSyncFrame frame) { return LipSync.TryProcessAnalyzerAtTime(handle, pcm, timeSeconds, out frame); }
        public bool ProcessMapped(float[] pcm, LipSyncMapperKind mapperKind, out LipSyncMappedFrame frame) { return LipSync.TryProcessAnalyzerMapped(handle, pcm, mapperKind, out frame); }
        public bool ProcessAtTimeMapped(float[] pcm, float timeSeconds, LipSyncMapperKind mapperKind, out LipSyncMappedFrame frame) { return LipSync.TryProcessAnalyzerAtTimeMapped(handle, pcm, timeSeconds, mapperKind, out frame); }
        public bool ProcessInterleaved(float[] pcm, int frameCount, int channels, out LipSyncFrame frame) { return LipSync.TryProcessAnalyzerInterleaved(handle, pcm, frameCount, channels, out frame); }
        public bool ProcessInterleavedAtTime(float[] pcm, int frameCount, int channels, float timeSeconds, out LipSyncFrame frame) { return LipSync.TryProcessAnalyzerInterleavedAtTime(handle, pcm, frameCount, channels, timeSeconds, out frame); }
        public bool ProcessInterleavedMapped(float[] pcm, int frameCount, int channels, LipSyncMapperKind mapperKind, out LipSyncMappedFrame frame) { return LipSync.TryProcessAnalyzerInterleavedMapped(handle, pcm, frameCount, channels, mapperKind, out frame); }
        public bool ProcessInterleavedAtTimeMapped(float[] pcm, int frameCount, int channels, float timeSeconds, LipSyncMapperKind mapperKind, out LipSyncMappedFrame frame) { return LipSync.TryProcessAnalyzerInterleavedAtTimeMapped(handle, pcm, frameCount, channels, timeSeconds, mapperKind, out frame); }
        public bool SetTimedCues(LipSyncTimedCue[] cues) { return LipSync.TrySetTimedCues(handle, cues); }
        public bool ClearTimedCues() { return LipSync.TryClearTimedCues(handle); }

        public void Dispose()
        {
            if (handle == IntPtr.Zero) return;
            LipSync.DestroyAnalyzer(handle);
            handle = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }

        ~LipSyncAnalyzer() { Dispose(); }
    }

    public static class LipSync
    {
        private const string NativeLibrary = "lip_sync";
        private static bool nativeAvailable = true;
        private static bool reportedNativeUnavailable;

        public static bool TryRecognizeVowel(float[] pcm, int sampleRate, out LipSyncVowel vowel)
        {
            vowel = default(LipSyncVowel);
            if (!nativeAvailable || pcm == null || pcm.Length == 0 || sampleRate <= 0) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.recognize_vowel(pcm, (UIntPtr)pcm.Length, (uint)sampleRate, out vowel); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#else
            ReportNativeUnavailable("lip_sync native plugin is only included for Windows x86_64.");
#endif
            return false;
        }

        public static bool TryGetDefaultOptions(int sampleRate, out LipSyncOptions options)
        {
            options = default(LipSyncOptions);
            if (!nativeAvailable || sampleRate <= 0) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try
            {
                options = NativeMethods.lipsync_default_options((uint)sampleRate);
                return options.sampleRate > 0;
            }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        public static bool TryGetSingingOptions(int sampleRate, out LipSyncOptions options)
        {
            options = default(LipSyncOptions);
            if (!nativeAvailable || sampleRate <= 0) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try
            {
                options = NativeMethods.lipsync_singing_options((uint)sampleRate);
                return options.sampleRate > 0;
            }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        public static bool TryCreateAnalyzer(int sampleRate, bool singingMode, out LipSyncAnalyzer analyzer)
        {
            LipSyncOptions options = LipSyncOptions.Create(sampleRate, singingMode, false, false, true);
            return TryCreateAnalyzer(options, out analyzer);
        }

        public static bool TryCreateAnalyzer(LipSyncOptions options, out LipSyncAnalyzer analyzer)
        {
            analyzer = null;
            if (!nativeAvailable || options.sampleRate == 0) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try
            {
                IntPtr handle = NativeMethods.lipsync_create_with_options(options);
                if (handle == IntPtr.Zero) return false;
                analyzer = new LipSyncAnalyzer(handle);
                return true;
            }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#else
            ReportNativeUnavailable("lip_sync native plugin is only included for Windows x86_64.");
#endif
            return false;
        }

        public static bool TryMapFrame(LipSyncFrame frame, LipSyncMapperKind mapperKind, out LipSyncMappedFrame mappedFrame)
        {
            mappedFrame = LipSyncMappedFrame.Empty;
            if (!nativeAvailable) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_map_frame(ref frame, (uint)mapperKind, out mappedFrame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryGetBestClass(ref LipSyncFrame frame, out LipSyncClass bestClass)
        {
            bestClass = LipSyncClass.Rest;
            if (!nativeAvailable) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try
            {
                uint classIndex;
                bool ok = NativeMethods.lipsync_frame_best_class(ref frame, out classIndex);
                if (!ok) return false;
                bestClass = (LipSyncClass)classIndex;
                return true;
            }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryGetLegacyVowel(ref LipSyncFrame frame, out LipSyncVowel vowel)
        {
            vowel = default(LipSyncVowel);
            if (!nativeAvailable) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try
            {
                int vowelIndex;
                bool ok = NativeMethods.lipsync_frame_legacy_vowel(ref frame, out vowelIndex);
                if (!ok || vowelIndex < 0 || vowelIndex > (int)LipSyncVowel.O) return false;
                vowel = (LipSyncVowel)vowelIndex;
                return true;
            }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryProcessAnalyzer(IntPtr handle, float[] pcm, out LipSyncFrame frame)
        {
            frame = default(LipSyncFrame);
            if (!nativeAvailable || handle == IntPtr.Zero || pcm == null || pcm.Length == 0) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_process(handle, pcm, (UIntPtr)pcm.Length, out frame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryProcessAnalyzerAtTime(IntPtr handle, float[] pcm, float timeSeconds, out LipSyncFrame frame)
        {
            frame = default(LipSyncFrame);
            if (!nativeAvailable || handle == IntPtr.Zero || pcm == null || pcm.Length == 0) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_process_at_time(handle, pcm, (UIntPtr)pcm.Length, Mathf.Max(0f, timeSeconds), out frame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryProcessAnalyzerMapped(IntPtr handle, float[] pcm, LipSyncMapperKind mapperKind, out LipSyncMappedFrame frame)
        {
            frame = LipSyncMappedFrame.Empty;
            if (!nativeAvailable || handle == IntPtr.Zero || pcm == null || pcm.Length == 0) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_process_mapped(handle, pcm, (UIntPtr)pcm.Length, (uint)mapperKind, out frame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryProcessAnalyzerAtTimeMapped(IntPtr handle, float[] pcm, float timeSeconds, LipSyncMapperKind mapperKind, out LipSyncMappedFrame frame)
        {
            frame = LipSyncMappedFrame.Empty;
            if (!nativeAvailable || handle == IntPtr.Zero || pcm == null || pcm.Length == 0) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_process_at_time_mapped(handle, pcm, (UIntPtr)pcm.Length, Mathf.Max(0f, timeSeconds), (uint)mapperKind, out frame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryProcessAnalyzerInterleaved(IntPtr handle, float[] pcm, int frameCount, int channels, out LipSyncFrame frame)
        {
            frame = default(LipSyncFrame);
            if (!CanProcessInterleaved(handle, pcm, frameCount, channels)) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_process_interleaved(handle, pcm, (UIntPtr)frameCount, (uint)channels, out frame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryProcessAnalyzerInterleavedAtTime(IntPtr handle, float[] pcm, int frameCount, int channels, float timeSeconds, out LipSyncFrame frame)
        {
            frame = default(LipSyncFrame);
            if (!CanProcessInterleaved(handle, pcm, frameCount, channels)) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_process_interleaved_at_time(handle, pcm, (UIntPtr)frameCount, (uint)channels, Mathf.Max(0f, timeSeconds), out frame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryProcessAnalyzerInterleavedMapped(IntPtr handle, float[] pcm, int frameCount, int channels, LipSyncMapperKind mapperKind, out LipSyncMappedFrame frame)
        {
            frame = LipSyncMappedFrame.Empty;
            if (!CanProcessInterleaved(handle, pcm, frameCount, channels)) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_process_interleaved_mapped(handle, pcm, (UIntPtr)frameCount, (uint)channels, (uint)mapperKind, out frame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryProcessAnalyzerInterleavedAtTimeMapped(IntPtr handle, float[] pcm, int frameCount, int channels, float timeSeconds, LipSyncMapperKind mapperKind, out LipSyncMappedFrame frame)
        {
            frame = LipSyncMappedFrame.Empty;
            if (!CanProcessInterleaved(handle, pcm, frameCount, channels)) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_process_interleaved_at_time_mapped(handle, pcm, (UIntPtr)frameCount, (uint)channels, Mathf.Max(0f, timeSeconds), (uint)mapperKind, out frame); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TrySetTimedCues(IntPtr handle, LipSyncTimedCue[] cues)
        {
            if (!nativeAvailable || handle == IntPtr.Zero) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try
            {
                if (cues == null || cues.Length == 0) return NativeMethods.lipsync_set_timed_cues(handle, null, UIntPtr.Zero);
                return NativeMethods.lipsync_set_timed_cues(handle, cues, (UIntPtr)cues.Length);
            }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static bool TryClearTimedCues(IntPtr handle)
        {
            if (!nativeAvailable || handle == IntPtr.Zero) return false;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { return NativeMethods.lipsync_clear_timed_cues(handle); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
            return false;
        }

        internal static void DestroyAnalyzer(IntPtr handle)
        {
            if (handle == IntPtr.Zero || !nativeAvailable) return;
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            try { NativeMethods.lipsync_destroy(handle); }
            catch (Exception exception) when (IsNativeException(exception)) { ReportNativeUnavailable(exception); }
#endif
        }

        private static bool CanProcessInterleaved(IntPtr handle, float[] pcm, int frameCount, int channels)
        {
            if (!nativeAvailable || handle == IntPtr.Zero || pcm == null || frameCount <= 0 || channels <= 0) return false;
            return pcm.Length >= frameCount * channels;
        }

        private static bool IsNativeException(Exception exception)
        {
            return exception is DllNotFoundException || exception is EntryPointNotFoundException || exception is BadImageFormatException;
        }

        private static void ReportNativeUnavailable(Exception exception) { ReportNativeUnavailable(exception.Message); }
        private static void ReportNativeUnavailable(string message)
        {
            nativeAvailable = false;
            if (reportedNativeUnavailable) return;
            reportedNativeUnavailable = true;
            Debug.LogWarning($"lip_sync native plugin is unavailable: {message}");
        }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        private static class NativeMethods
        {
            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool recognize_vowel([In] float[] pcmData, UIntPtr len, uint sampleRate, out LipSyncVowel result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            internal static extern LipSyncOptions lipsync_default_options(uint sampleRate);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            internal static extern LipSyncOptions lipsync_singing_options(uint sampleRate);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_frame_best_class([In] ref LipSyncFrame frame, out uint result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_frame_legacy_vowel([In] ref LipSyncFrame frame, out int result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_map_frame([In] ref LipSyncFrame frame, uint mapperKind, out LipSyncMappedFrame result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            internal static extern IntPtr lipsync_create_with_options(LipSyncOptions options);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_process(IntPtr analyzer, [In] float[] pcmData, UIntPtr len, out LipSyncFrame result);


            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_process_at_time(IntPtr analyzer, [In] float[] pcmData, UIntPtr len, float timeSeconds, out LipSyncFrame result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_process_mapped(IntPtr analyzer, [In] float[] pcmData, UIntPtr len, uint mapperKind, out LipSyncMappedFrame result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_process_at_time_mapped(IntPtr analyzer, [In] float[] pcmData, UIntPtr len, float timeSeconds, uint mapperKind, out LipSyncMappedFrame result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_process_interleaved(IntPtr analyzer, [In] float[] pcmData, UIntPtr frameCount, uint channels, out LipSyncFrame result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_process_interleaved_at_time(IntPtr analyzer, [In] float[] pcmData, UIntPtr frameCount, uint channels, float timeSeconds, out LipSyncFrame result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_process_interleaved_mapped(IntPtr analyzer, [In] float[] pcmData, UIntPtr frameCount, uint channels, uint mapperKind, out LipSyncMappedFrame result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_process_interleaved_at_time_mapped(IntPtr analyzer, [In] float[] pcmData, UIntPtr frameCount, uint channels, float timeSeconds, uint mapperKind, out LipSyncMappedFrame result);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_set_timed_cues(IntPtr analyzer, [In] LipSyncTimedCue[] cues, UIntPtr len);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            [return: MarshalAs(UnmanagedType.I1)]
            internal static extern bool lipsync_clear_timed_cues(IntPtr analyzer);

            [DllImport(NativeLibrary, CallingConvention = CallingConvention.Cdecl)]
            internal static extern void lipsync_destroy(IntPtr analyzer);
        }
#endif
    }
}