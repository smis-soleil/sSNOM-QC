import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Keep matplotlib/font cache writes inside writable temp dirs during tests.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import app as snom_app


class SessionState(dict):
    """Minimal attr+dict compatible session state for tests."""

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyColumn(DummyContext):
    pass


class DummyBox:
    def __init__(self):
        self.emptied = False

    def empty(self):
        self.emptied = True


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.messages = []
        self.markdown_calls = []
        self.button_calls = []
        self.file_uploader_value = None
        self.segmented_control_value = snom_app.DEMOD_OPTIONS[0]
        self.button_value = False
        self.button_values = {}
        self.text_input_values = {}
        self.rerun_called = False

    @property
    def sidebar(self):
        return DummyContext()

    def spinner(self, _msg):
        return DummyContext()

    def warning(self, msg):
        self.messages.append(("warning", msg))

    def success(self, msg):
        self.messages.append(("success", msg))

    def error(self, msg):
        self.messages.append(("error", msg))

    def info(self, msg):
        self.messages.append(("info", msg))

    def title(self, _msg):
        return None

    def write(self, _msg):
        return None

    def divider(self):
        return None

    def file_uploader(self, *_args, **_kwargs):
        return self.file_uploader_value

    def markdown(self, html, unsafe_allow_html=False):
        self.markdown_calls.append((html, unsafe_allow_html))
        return DummyBox()

    def segmented_control(self, *_args, **_kwargs):
        return self.segmented_control_value

    def text_input(self, _label, key=None, value="", **_kwargs):
        if key is None:
            return value
        if key in self.text_input_values:
            self.session_state[key] = self.text_input_values[key]
        elif key not in self.session_state:
            self.session_state[key] = value
        return self.session_state[key]

    def button(self, label, *_args, **kwargs):
        self.button_calls.append(label)
        key = kwargs.get("key")
        if key in self.button_values:
            return self.button_values[key]
        return self.button_values.get(label, self.button_value)

    def rerun(self):
        self.rerun_called = True

    def columns(self, n):
        if isinstance(n, int):
            count = n
        else:
            count = len(n)
        return [DummyColumn() for _ in range(count)]

    def pyplot(self, *_args, **_kwargs):
        return None

    def expander(self, *_args, **_kwargs):
        return DummyContext()

    def exception(self, _e):
        return None


class UploadedFile:
    def __init__(self, name, payload=b"content"):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload


def make_measparams(project="p"):
    return {
        "Project": project,
        "Date": "2024-03-31",
        "TipAmplitude": 70,
        "Averaging": 16,
        "Integrationtime": 20,
        "InterferometerCenterDistance": [1, 2],
    }


def make_file_data(name, phase=0.0):
    wn = np.linspace(600.0, 1800.0, 500)
    base = 1.0 + 0.01 * np.sin(np.linspace(0 + phase, 10 + phase, 500))
    data = {
        "Wavenumber": wn,
        "O2A": base,
        "O3A": base * 1.01,
        "O4A": base * 1.02,
        "O5A": base * 1.03,
    }
    return {"name": name, "data": data, "measparams": make_measparams(project=name)}


class TestCalculations(unittest.TestCase):
    def test_calculate_snr_stats_expected_values(self):
        wn = np.array([700, 800, 900, 1000, 1100], dtype=float)
        ratio = np.array([2.0, 2.2, 2.4, 2.6, 2.8], dtype=float)

        stats = snom_app.calculate_snr_stats(wn, ratio, 800, 1100)

        np.testing.assert_allclose(stats["snr"], np.mean(ratio[1:]) / np.std(ratio[1:]))
        np.testing.assert_allclose(stats["y_min"], np.min(ratio[1:]) - np.std(ratio[1:]))
        np.testing.assert_allclose(stats["y_max"], np.max(ratio[1:]) + np.std(ratio[1:]))

    def test_compute_plot_data_expected_structure(self):
        rng = np.random.default_rng(1)
        wn = np.linspace(650.0, 1800.0, 1024)
        sp1 = 2.0 + rng.normal(0.0, 0.01, 1024)
        sp2 = 1.0 + rng.normal(0.0, 0.01, 1024)

        ratio, s1, s2 = snom_app.compute_plot_data.__wrapped__(wn, sp1, wn, sp2)
        self.assertEqual(ratio.shape, wn.shape)
        self.assertIn("snr", s1)
        self.assertIn("snr", s2)

    def test_compute_raises_on_mismatched_lengths(self):
        wn = np.linspace(650.0, 1800.0, 10)
        sp1 = np.ones(10)
        sp2 = np.ones(9)

        with self.assertRaises(ValueError):
            snom_app.compute_plot_data.__wrapped__(wn, sp1, wn[:-1], sp2)

    def test_calculate_snr_stats_raises_for_empty_range(self):
        wn = np.array([700.0, 800.0, 900.0])
        ratio = np.array([1.0, 1.1, 1.2])
        with self.assertRaises(ValueError):
            snom_app.calculate_snr_stats(wn, ratio, 1500.0, 1600.0)

    def test_parse_custom_snr_ranges(self):
        fake_st = FakeStreamlit()
        fake_st.session_state["custom_snr_start_0"] = "800"
        fake_st.session_state["custom_snr_end_0"] = "1300"
        fake_st.session_state["custom_snr_start_1"] = "abc"
        fake_st.session_state["custom_snr_end_1"] = "1600"
        wn = np.linspace(700.0, 1500.0, 100)

        with patch.object(snom_app, "st", fake_st):
            ranges, errors = snom_app.parse_custom_snr_ranges(2, wn_reference=wn)

        self.assertEqual(ranges, [(800.0, 1300.0)])
        self.assertEqual(len(errors), 1)

    def test_parse_custom_snr_ranges_reports_invalid_rows(self):
        fake_st = FakeStreamlit()
        fake_st.session_state["custom_snr_start_0"] = "900"
        fake_st.session_state["custom_snr_end_0"] = ""
        fake_st.session_state["custom_snr_start_1"] = "1200"
        fake_st.session_state["custom_snr_end_1"] = "1000"
        wn = np.linspace(700.0, 1500.0, 100)

        with patch.object(snom_app, "st", fake_st):
            ranges, errors = snom_app.parse_custom_snr_ranges(2, wn_reference=wn)

        self.assertEqual(ranges, [])
        self.assertEqual(len(errors), 2)

    def test_assess_file_compatibility_detects_point_and_range_mismatch(self):
        f1 = make_file_data("a")
        f2 = make_file_data("b")
        f2["data"]["Wavenumber"] = np.linspace(500.0, 1700.0, 400)

        warnings, preset_valid = snom_app.assess_file_compatibility(f1, f2)

        self.assertTrue(preset_valid)
        self.assertTrue(any("different numbers of points" in w for w in warnings))
        self.assertTrue(any("different wavenumber ranges" in w for w in warnings))

    def test_assess_file_compatibility_detects_invalid_preset_ranges(self):
        wn_small = np.linspace(3000.0, 3100.0, 50)
        f1 = make_file_data("a")
        f2 = make_file_data("b")
        f1["data"]["Wavenumber"] = wn_small
        f2["data"]["Wavenumber"] = wn_small.copy()

        warnings, preset_valid = snom_app.assess_file_compatibility(f1, f2)
        self.assertFalse(preset_valid)
        self.assertTrue(any("Preset SNR ranges are not fully covered" in w for w in warnings))


class TestFileAndReader(unittest.TestCase):
    def test_temp_file_context_creates_and_removes_file(self):
        uploaded = UploadedFile("test.txt", b"abc123")

        with snom_app.temp_file_context(uploaded) as temp_path:
            p = Path(temp_path)
            self.assertTrue(p.exists())
            self.assertEqual(p.read_bytes(), b"abc123")

        self.assertFalse(Path(temp_path).exists())

    def test_load_nea_core_returns_reader_output_and_cleans_temp(self):
        captured = {}

        class FakeReader:
            def __init__(self, path):
                captured["path"] = path

            def read(self):
                return {"O2A": np.array([1.0]), "Wavenumber": np.array([1000.0])}, {"Project": "x"}

        with patch.object(snom_app.readers, "NeaSpectralReader", FakeReader):
            data, meas = snom_app.load_nea.__wrapped__("x.txt", b"payload")

        self.assertIn("O2A", data)
        self.assertEqual(meas["Project"], "x")
        self.assertFalse(Path(captured["path"]).exists())


class TestPlotting(unittest.TestCase):
    def test_add_figure_caption_adds_text(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        before = len(fig.texts)
        f1 = make_file_data("a")
        f2 = make_file_data("b")

        snom_app.add_figure_caption(fig, f1, f2)

        self.assertEqual(len(fig.texts), before + 1)
        self.assertIn("Exp. Date", fig.texts[-1].get_text())
        plt.close(fig)

    def test_create_comparison_plot_returns_expected_axes(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.show_motd = True

        with patch.object(snom_app, "st", fake_st):
            fig = snom_app.create_comparison_plot(make_file_data("a"), make_file_data("b", phase=0.5), "O2A")

        self.assertEqual(len(fig.axes), 3)
        self.assertFalse(fake_st.session_state.show_motd)
        snom_app.plt.close(fig)

    def test_create_comparison_plot_raises_on_mismatch(self):
        f1 = make_file_data("a")
        f2 = make_file_data("b")
        f2["data"]["O2A"] = f2["data"]["O2A"][:-1]

        with self.assertRaises(ValueError):
            snom_app.create_comparison_plot(f1, f2, "O2A")

    def test_create_custom_snr_plot_returns_expected_axes(self):
        fig = snom_app.create_custom_snr_plot(
            make_file_data("a"),
            make_file_data("b", phase=0.2),
            "O2A",
            [(800.0, 1300.0), (900.0, 1200.0)],
        )
        self.assertEqual(len(fig.axes), 2)
        snom_app.plt.close(fig)


class TestStateAndUIFlows(unittest.TestCase):
    def test_init_and_reset_session_state(self):
        fake_st = FakeStreamlit()

        with patch.object(snom_app, "st", fake_st):
            snom_app.init_session_state()
            self.assertEqual(fake_st.session_state.uploaded_files, [])
            self.assertEqual(fake_st.session_state.upload_widget_key, 0)
            self.assertTrue(fake_st.session_state.show_motd)
            self.assertEqual(fake_st.session_state.perf_stats, {})
            self.assertEqual(fake_st.session_state.snr_row_count, 1)
            self.assertEqual(fake_st.session_state.custom_snr_ranges, [])
            self.assertTrue(fake_st.session_state.default_snr_valid)

            fake_st.session_state.uploaded_files = [1]
            fake_st.session_state.upload_widget_key = 3
            fake_st.session_state.show_motd = False
            fake_st.session_state.snr_row_count = 4
            fake_st.session_state.custom_snr_ranges = [(1, 2)]
            fake_st.session_state["custom_snr_start_0"] = "1"
            fake_st.session_state["custom_snr_end_0"] = "2"
            snom_app.reset_app()

        self.assertEqual(fake_st.session_state.uploaded_files, [])
        self.assertEqual(fake_st.session_state.upload_widget_key, 4)
        self.assertTrue(fake_st.session_state.show_motd)
        self.assertEqual(fake_st.session_state.snr_row_count, 1)
        self.assertEqual(fake_st.session_state.custom_snr_ranges, [])
        self.assertEqual(fake_st.session_state["custom_snr_start_0"], "")
        self.assertEqual(fake_st.session_state["custom_snr_end_0"], "")
        self.assertTrue(fake_st.session_state.default_snr_valid)

    def test_handle_file_upload_success(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = []
        fake_st.session_state.upload_widget_key = 0
        fake_st.session_state.perf_stats = {}

        upload = UploadedFile("f1.txt", b"x")

        with patch.object(snom_app, "st", fake_st), patch.object(
            snom_app, "load_nea", return_value=({"Wavenumber": np.array([1.0]), "O2A": np.array([1.0])}, make_measparams())
        ):
            ok = snom_app.handle_file_upload(upload)

        self.assertTrue(ok)
        self.assertEqual(len(fake_st.session_state.uploaded_files), 1)
        self.assertEqual(fake_st.session_state.upload_widget_key, 1)
        self.assertIn("load:f1.txt", fake_st.session_state.perf_stats)

    def test_handle_file_upload_reports_loader_exception(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = []
        fake_st.session_state.upload_widget_key = 0
        fake_st.session_state.perf_stats = {}

        with patch.object(snom_app, "st", fake_st), patch.object(
            snom_app, "load_nea", side_effect=RuntimeError("reader failure")
        ):
            ok = snom_app.handle_file_upload(UploadedFile("f1.txt"))

        self.assertFalse(ok)
        self.assertTrue(any("Error loading file" in msg for t, msg in fake_st.messages if t == "error"))

    def test_handle_file_upload_duplicate_and_max_files(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = [{"name": "f1.txt"}, {"name": "f2.txt"}]
        fake_st.session_state.upload_widget_key = 0

        with patch.object(snom_app, "st", fake_st):
            self.assertFalse(snom_app.handle_file_upload(UploadedFile("f1.txt")))
            self.assertFalse(snom_app.handle_file_upload(UploadedFile("f3.txt")))

        warnings = [m for t, m in fake_st.messages if t == "warning"]
        self.assertTrue(any("Maximum" in w for w in warnings))

    def test_render_sidebar_returns_order(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = [make_file_data("a"), make_file_data("b")]
        fake_st.session_state.upload_widget_key = 0

        with patch.object(snom_app, "st", fake_st):
            order = snom_app.render_sidebar()

        self.assertEqual(order, snom_app.DEMOD_OPTIONS[0])
        self.assertTrue(fake_st.session_state.default_snr_valid)

    def test_render_sidebar_reset_triggers_rerun(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = []
        fake_st.session_state.upload_widget_key = 0
        fake_st.button_values["Reset All"] = True

        with patch.object(snom_app, "st", fake_st), patch.object(snom_app, "reset_app") as reset_mock:
            _order = snom_app.render_sidebar()

        reset_mock.assert_called_once()
        self.assertTrue(fake_st.rerun_called)

    def test_render_sidebar_calculate_sets_custom_ranges(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = [make_file_data("a"), make_file_data("b")]
        fake_st.session_state.upload_widget_key = 0
        fake_st.session_state.snr_row_count = 1
        fake_st.text_input_values["custom_snr_start_0"] = "780"
        fake_st.text_input_values["custom_snr_end_0"] = "1260"
        fake_st.button_values["Display graphs"] = True

        with patch.object(snom_app, "st", fake_st):
            _order = snom_app.render_sidebar()

        self.assertEqual(fake_st.session_state.custom_snr_ranges, [(780.0, 1260.0)])

    def test_render_sidebar_calculate_rejects_out_of_data_range(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = [make_file_data("a"), make_file_data("b")]
        fake_st.session_state.upload_widget_key = 0
        fake_st.session_state.snr_row_count = 1
        fake_st.text_input_values["custom_snr_start_0"] = "4000"
        fake_st.text_input_values["custom_snr_end_0"] = "4100"
        fake_st.button_values["Display graphs"] = True

        with patch.object(snom_app, "st", fake_st):
            _order = snom_app.render_sidebar()

        self.assertEqual(fake_st.session_state.custom_snr_ranges, [])
        self.assertTrue(any("no data points found" in m for t, m in fake_st.messages if t == "error"))

    def test_render_custom_snr_controls_requires_data_for_display(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.snr_row_count = 1
        fake_st.button_values["Display graphs"] = True

        with patch.object(snom_app, "st", fake_st):
            snom_app.render_custom_snr_controls(wn_reference=None)

        self.assertEqual(fake_st.session_state.custom_snr_ranges, [])
        self.assertTrue(
            any("Upload two valid spectra" in msg for t, msg in fake_st.messages if t == "error")
        )

    def test_render_sidebar_plus_adds_new_range_row(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = [make_file_data("a"), make_file_data("b")]
        fake_st.session_state.upload_widget_key = 0
        fake_st.session_state.snr_row_count = 1
        fake_st.button_values["custom_snr_add_btn"] = True

        with patch.object(snom_app, "st", fake_st):
            _order = snom_app.render_sidebar()

        self.assertEqual(fake_st.session_state.snr_row_count, 2)
        self.assertTrue(fake_st.rerun_called)

    def test_render_sidebar_minus_removes_last_range_row(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = [make_file_data("a"), make_file_data("b")]
        fake_st.session_state.upload_widget_key = 0
        fake_st.session_state.snr_row_count = 2
        fake_st.session_state["custom_snr_start_1"] = "900"
        fake_st.session_state["custom_snr_end_1"] = "1200"
        fake_st.button_values["custom_snr_remove_btn"] = True

        with patch.object(snom_app, "st", fake_st):
            _order = snom_app.render_sidebar()

        self.assertEqual(fake_st.session_state.snr_row_count, 1)
        self.assertNotIn("custom_snr_start_1", fake_st.session_state)
        self.assertNotIn("custom_snr_end_1", fake_st.session_state)
        self.assertTrue(fake_st.rerun_called)

    def test_render_sidebar_no_minus_when_single_row(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = [make_file_data("a"), make_file_data("b")]
        fake_st.session_state.upload_widget_key = 0
        fake_st.session_state.snr_row_count = 1

        with patch.object(snom_app, "st", fake_st):
            _order = snom_app.render_sidebar()

        self.assertNotIn(snom_app.REMOVE_ROW_BUTTON_LABEL, fake_st.button_calls)

    def test_render_sidebar_warns_for_incompatible_files(self):
        fake_st = FakeStreamlit()
        f1 = make_file_data("a")
        f2 = make_file_data("b")
        f2["data"]["Wavenumber"] = np.linspace(500.0, 1700.0, 400)
        fake_st.session_state.uploaded_files = [f1, f2]
        fake_st.session_state.upload_widget_key = 0

        with patch.object(snom_app, "st", fake_st):
            _order = snom_app.render_sidebar()

        warning_msgs = [m for t, m in fake_st.messages if t == "warning"]
        self.assertTrue(any("might not be compatible" in m for m in warning_msgs))
        self.assertTrue(fake_st.session_state.default_snr_valid)

    def test_render_metadata_outputs_two_markdowns(self):
        fake_st = FakeStreamlit()
        fake_st.session_state.uploaded_files = [make_file_data("a"), make_file_data("b")]

        with patch.object(snom_app, "st", fake_st):
            snom_app.render_metadata()

        # One heading markdown and two metadata blocks.
        self.assertGreaterEqual(len(fake_st.markdown_calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
