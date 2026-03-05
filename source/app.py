import os
import datetime
import tempfile
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from contextlib import contextmanager

from pySNOM import readers
from matplotlib.ticker import FormatStrFormatter


# Configuration constants
START_WN1, END_WN1 = 800, 1300
START_WN2, END_WN2 = 650, 1800
DEMOD_OPTIONS = ["O2A", "O3A", "O4A", "O5A"]
MAX_FILES = 2
ADD_ROW_BUTTON_LABEL = "\u2795"
REMOVE_ROW_BUTTON_LABEL = "\u2796"

# Set page config first
st.set_page_config(layout="wide", page_title="sSNOM-QC")


@contextmanager
def temp_file_context(uploaded_file):
    """Context manager for temporary file handling."""
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=os.path.splitext(uploaded_file.name)[1]
    )
    try:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        yield tmp_file.name
    finally:
        os.unlink(tmp_file.name)


@st.cache_data(show_spinner=False)
def load_nea(file_name, file_bytes):
    """Load NeaSNOM data with caching."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        nea_reader = readers.NeaSpectralReader(tmp_file_path)
        nea_data, nea_measparams = nea_reader.read()
        return nea_data, nea_measparams
    finally:
        os.unlink(tmp_file_path)


def init_session_state():
    """Initialize session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'upload_widget_key' not in st.session_state:
        st.session_state.upload_widget_key = 0
    if 'show_motd' not in st.session_state:
        st.session_state.show_motd = True
    if 'perf_stats' not in st.session_state:
        st.session_state.perf_stats = {}
    if 'snr_row_count' not in st.session_state:
        st.session_state.snr_row_count = 1
    if 'custom_snr_ranges' not in st.session_state:
        st.session_state.custom_snr_ranges = []
    if 'default_snr_valid' not in st.session_state:
        st.session_state.default_snr_valid = True


def get_wavenumber_from_file(file_data):
    """Extract wavenumber array from uploaded file dict."""
    data = file_data.get("data", {}) if isinstance(file_data, dict) else {}
    if isinstance(data, dict):
        return data.get("Wavenumber")
    return None


def assess_file_compatibility(file_data_1, file_data_2):
    """Return warnings and default SNR validity for the two uploaded files."""
    warnings = []
    preset_valid = True

    wn1 = get_wavenumber_from_file(file_data_1)
    wn2 = get_wavenumber_from_file(file_data_2)
    if wn1 is None or wn2 is None:
        return warnings, preset_valid

    if len(wn1) != len(wn2):
        warnings.append(
            "Files have different numbers of points and might not be compatible."
        )

    range_equal = np.isclose(np.min(wn1), np.min(wn2)) and np.isclose(np.max(wn1), np.max(wn2))
    if not range_equal:
        warnings.append(
            "Files have different wavenumber ranges and might not be compatible."
        )

    default_ranges = [(START_WN1, END_WN1), (START_WN2, END_WN2)]
    for start_wn, end_wn in default_ranges:
        has_data_1 = np.any((wn1 >= start_wn) & (wn1 <= end_wn))
        has_data_2 = np.any((wn2 >= start_wn) & (wn2 <= end_wn))
        if not has_data_1 or not has_data_2:
            preset_valid = False

    if not preset_valid:
        warnings.append(
            "Preset SNR ranges are not fully covered by uploaded data. Main SNR graphs are disabled."
        )

    return warnings, preset_valid



def reset_custom_snr_controls():
    """Reset custom SNR controls and calculated ranges."""
    st.session_state.snr_row_count = 1
    st.session_state.custom_snr_ranges = []
    for key in list(st.session_state.keys()):
        if key.startswith("custom_snr_start_") or key.startswith("custom_snr_end_"):
            st.session_state.pop(key, None)
    st.session_state["custom_snr_start_0"] = ""
    st.session_state["custom_snr_end_0"] = ""


def reset_app():
    """Reset application state."""
    st.session_state.uploaded_files = []
    st.session_state.upload_widget_key += 1
    st.session_state.show_motd = True
    st.session_state.default_snr_valid = True
    reset_custom_snr_controls()



def handle_file_upload(uploaded_file):
    """Handle file upload with validation."""
    if len(st.session_state.uploaded_files) >= MAX_FILES:
        st.warning(f"Maximum of {MAX_FILES} files allowed. Use reset button to clear.")
        return False
    
    # Check if file already uploaded
    if any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
        return False
    
    try:
        t0 = time.perf_counter()
        with st.spinner(f"Loading {uploaded_file.name}..."):
            data, measparams = load_nea(uploaded_file.name, uploaded_file.getvalue())
            
            st.session_state.uploaded_files.append({
                'name': uploaded_file.name,
                'data': data,
                'measparams': measparams
            })
            
            # Increment key to clear the uploader
            st.session_state.upload_widget_key += 1
        st.session_state.perf_stats[f"load:{uploaded_file.name}"] = time.perf_counter() - t0
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return False


def calculate_snr_stats(wn, ratio, start_wn, end_wn):
    """Calculate SNR and plot limits for a given range."""
    mask = (wn >= start_wn) & (wn <= end_wn)
    ratio_slice = ratio[mask]
    if ratio_slice.size == 0:
        raise ValueError(f"No data points in selected range [{start_wn}, {end_wn}].")
    mean_ratio = np.mean(ratio_slice)
    std_ratio = np.std(ratio_slice)
    
    return {
        'snr': mean_ratio / std_ratio,
        'y_min': np.min(ratio_slice) - std_ratio,
        'y_max': np.max(ratio_slice) + std_ratio
    }


@st.cache_resource
def setup_plot_style():
    """Setup matplotlib style (cached to avoid reloading)."""
    try:
        plt.style.use("source/plot-style.mplstyle")
    except:
        pass  # Use default style if custom not found


@st.cache_data(show_spinner=False)
def compute_plot_data(wn1, sp1, wn2, sp2):
    """Compute cached ratio and SNR stats for plotting."""
    # Validate spectra length
    if len(sp1) != len(sp2):
        raise ValueError("Spectra must have the same length.")
    if len(wn1) != len(wn2):
        raise ValueError("Wavenumber arrays must have the same length.")
    
    ratio = sp1 / sp2

    stats1 = calculate_snr_stats(wn1, ratio, START_WN1, END_WN1)
    stats2 = calculate_snr_stats(wn1, ratio, START_WN2, END_WN2)

    return ratio, stats1, stats2


def create_comparison_plot(file_data_1, file_data_2, order):
    """Create the comparison plot figure."""
    sp1 = file_data_1['data'][order]
    wn1 = file_data_1['data']["Wavenumber"]
    sp2 = file_data_2['data'][order]
    wn2 = file_data_2['data']["Wavenumber"]
    ratio, stats1, stats2 = compute_plot_data(wn1, sp1, wn2, sp2)
    
    # Create figure
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.7, 0.3], height_ratios=[1, 1])
    
    # Left subplot: full spectra
    ax_left = plt.subplot(gs[:, 0])
    ax_left.plot(wn1, sp1, label=f"{file_data_1['name']} {order}")
    ax_left.plot(wn2, sp2, label=f"{file_data_2['name']} {order}")
    ax_left.set_xlim(0, 5000)
    ax_left.set_xlabel("Frequency / cm⁻¹")
    ax_left.set_ylabel(f"{order} / a.u.")
    ax_left.legend(loc="upper right")
    
    now_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
    ax_left.set_title(
        f"Project: {file_data_1['measparams']['Project']}\nPlot date: {now_str}",
        loc="left"
    )
    
    # Top-right subplot: first range
    ax_right_top = plt.subplot(gs[0, 1])
    ax_right_top.plot(wn1, ratio, label=f"SNR: {stats1['snr']:.1f}", color="#28ad2c")
    ax_right_top.set_xlim(START_WN1, END_WN1)
    ax_right_top.set_ylim(stats1['y_min'], stats1['y_max'])
    ax_right_top.set_ylabel(f"{order} Ratio / a.u.")
    ax_right_top.legend(loc="upper right")
    ax_right_top.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Bottom-right subplot: second range
    ax_right_bottom = plt.subplot(gs[1, 1])
    ax_right_bottom.plot(wn1, ratio, label=f"SNR: {stats2['snr']:.1f}", color="#e0147a")
    ax_right_bottom.set_xlim(START_WN2, END_WN2)
    ax_right_bottom.set_ylim(stats2['y_min'], stats2['y_max'])
    ax_right_bottom.set_xlabel("Frequency / cm⁻¹")
    ax_right_bottom.set_ylabel(f"{order} Ratio / a.u.")
    ax_right_bottom.legend(loc="upper right")
    ax_right_bottom.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Add figure caption
    add_figure_caption(fig, file_data_1, file_data_2)
    
    st.session_state.show_motd = False

    return fig


def create_custom_snr_plot(file_data_1, file_data_2, order, custom_ranges):
    """Create extra SNR plots for user-provided ranges."""
    sp1 = file_data_1['data'][order]
    wn1 = file_data_1['data']["Wavenumber"]
    sp2 = file_data_2['data'][order]
    wn2 = file_data_2['data']["Wavenumber"]
    ratio, _, _ = compute_plot_data(wn1, sp1, wn2, sp2)

    fig, axes = plt.subplots(len(custom_ranges), 1, figsize=(8, 3 * len(custom_ranges)))
    if len(custom_ranges) == 1:
        axes = [axes]

    colors = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#e69f00"]
    for idx, (ax, (start_wn, end_wn)) in enumerate(zip(axes, custom_ranges)):
        stats = calculate_snr_stats(wn1, ratio, start_wn, end_wn)
        ax.plot(wn1, ratio, color=colors[idx % len(colors)], label=f"SNR: {stats['snr']:.1f}")
        ax.set_xlim(start_wn, end_wn)
        ax.set_ylim(stats['y_min'], stats['y_max'])
        ax.set_ylabel(f"{order} Ratio / a.u.")
        ax.set_title(f"Custom SNR range: {start_wn:g} - {end_wn:g} cm⁻¹", loc="left")
        ax.legend(loc="upper right")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axes[-1].set_xlabel("Frequency / cm⁻¹")
    fig.tight_layout()
    return fig


def parse_custom_snr_ranges(row_count, wn_reference=None):
    """Parse user-provided custom SNR ranges from session state."""
    ranges = []
    errors = []

    for idx in range(row_count):
        start_raw = str(st.session_state.get(f"custom_snr_start_{idx}", "")).strip()
        end_raw = str(st.session_state.get(f"custom_snr_end_{idx}", "")).strip()

        if not start_raw and not end_raw:
            continue
        if not start_raw or not end_raw:
            errors.append(f"Row {idx + 1}: both start and end are required.")
            continue

        try:
            start_wn = float(start_raw)
            end_wn = float(end_raw)
        except ValueError:
            errors.append(f"Row {idx + 1}: start and end must be numeric.")
            continue

        if start_wn >= end_wn:
            errors.append(f"Row {idx + 1}: start must be lower than end.")
            continue

        if wn_reference is not None:
            range_has_data = np.any((wn_reference >= start_wn) & (wn_reference <= end_wn))
            if not range_has_data:
                errors.append(
                    f"Row {idx + 1}: no data points found in selected range [{start_wn}, {end_wn}]."
                )
                continue

        ranges.append((start_wn, end_wn))

    return ranges, errors


def render_custom_snr_controls(wn_reference=None):
    """Render custom SNR controls in a sidebar dropdown."""
    if 'snr_row_count' not in st.session_state:
        st.session_state.snr_row_count = 1
    if 'custom_snr_ranges' not in st.session_state:
        st.session_state.custom_snr_ranges = []

    st.markdown(
        """
        <style>
        .st-key-custom_snr_display_graphs_btn button,
        .st-key-custom_snr_add_btn button {
            background-color: #2e7d32 !important;
            color: white !important;
            border-color: #2e7d32 !important;
        }
        .st-key-custom_snr_remove_btn button {
            background-color: #c62828 !important;
            color: white !important;
            border-color: #c62828 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Custom SNR graphs"):
        row_count = st.session_state.snr_row_count

        header_cols = st.columns([0.6, 0.6, 0.4, 0.4])
        with header_cols[0]:
            st.markdown("**start**")
        with header_cols[1]:
            st.markdown("**end**")

        for idx in range(row_count):
            row_cols = st.columns([0.6, 0.6, 0.4, 0.4])
            with row_cols[0]:
                st.text_input(
                    f"start_{idx}",
                    key=f"custom_snr_start_{idx}",
                    label_visibility="collapsed"
                )
            with row_cols[1]:
                st.text_input(
                    f"end_{idx}",
                    key=f"custom_snr_end_{idx}",
                    label_visibility="collapsed"
                )
            with row_cols[2]:
                if idx == row_count - 1:
                    add_clicked = st.button(ADD_ROW_BUTTON_LABEL, key="custom_snr_add_btn", use_container_width=True)
                    if add_clicked:
                        st.session_state.snr_row_count += 1
                        st.rerun()
            with row_cols[3]:
                if idx == row_count - 1 and row_count > 1:
                    remove_clicked = st.button(REMOVE_ROW_BUTTON_LABEL, key="custom_snr_remove_btn", use_container_width=True)
                    if remove_clicked:
                        st.session_state.snr_row_count -= 1
                        st.session_state.pop(f"custom_snr_start_{idx}", None)
                        st.session_state.pop(f"custom_snr_end_{idx}", None)
                        st.rerun()

        calculate_clicked = st.button("Display graphs", key="custom_snr_display_graphs_btn", use_container_width=True)

        if calculate_clicked:
            if wn_reference is None:
                ranges = []
                errors = ["Upload two valid spectra before calculating custom SNR ranges."]
            else:
                ranges, errors = parse_custom_snr_ranges(st.session_state.snr_row_count, wn_reference)
            st.session_state.custom_snr_ranges = ranges
            for err in errors:
                st.error(err)
            if not errors and not ranges:
                st.warning("No valid custom SNR ranges entered.")


def add_figure_caption(fig, file_data_1, file_data_2):
    """Add caption with measurement parameters."""
    def format_params(params):
        return (
            f"Exp. Date: {params['Date']}" 
            f"TA: {params['TipAmplitude']} nm - Avg: {params['Averaging']} - "
            f"Int time: {params['Integrationtime']} ms - "
            f"Interferometer: {params['InterferometerCenterDistance'][0]}, "
            f"{params['InterferometerCenterDistance'][1]}"
        )
    
    caption = (
        f"{file_data_1['name']}\n  {format_params(file_data_1['measparams'])}\n\n"
        f"{file_data_2['name']}\n  {format_params(file_data_2['measparams'])}"
    )
    
    fig.text(0.1, -0.15, caption, ha='left', va='bottom', fontsize=12)


def render_sidebar():
    """Render sidebar UI."""
    st.title("sSNOM-QC")
    st.write("Upload two NeaSNOM files.")
    wn_reference = None
    
    # File uploader
    uploaded_file = st.file_uploader(
        "uploader",
        type=["txt"],
        key=f"uploader_{st.session_state.upload_widget_key}",
        label_visibility="collapsed"
    )
    
    # Handle upload
    if uploaded_file is not None:
        if handle_file_upload(uploaded_file):
            st.rerun()
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.markdown("### Uploaded Files:")
        for i, file_info in enumerate(st.session_state.uploaded_files, 1):
            st.write(f"**{i}.** {file_info['name']}")
    
    # Status and controls
    num_files = len(st.session_state.uploaded_files)
    
    if num_files == MAX_FILES:
        st.success("Both files uploaded.")
        order = st.segmented_control(
            "Select demodulation order",
            DEMOD_OPTIONS,
            selection_mode="single",
            width="content",
            default=DEMOD_OPTIONS[0]
        )
        first_data = st.session_state.uploaded_files[0].get("data", {})
        if isinstance(first_data, dict) and "Wavenumber" in first_data:
            wn_reference = first_data["Wavenumber"]

        compatibility_warnings, preset_valid = assess_file_compatibility(
            st.session_state.uploaded_files[0],
            st.session_state.uploaded_files[1],
        )
        st.session_state.default_snr_valid = preset_valid
        for warning_msg in compatibility_warnings:
            st.warning(warning_msg)
    elif num_files == 1:
        st.info("Upload one more file to proceed.")
        order = None
        st.session_state.default_snr_valid = True
    else:
        st.info("Upload your first spectrum file.")
        order = None
        st.session_state.default_snr_valid = True

    render_custom_snr_controls(wn_reference)
    
    # Reset button
    if st.button("Reset All", type="primary", use_container_width=True):
        reset_app()
        st.rerun()
    
    return order


def render_metadata():
    """Render metadata section."""
    st.divider()
    st.write("### Full metadata")
    
    col1, col2 = st.columns(2)
    
    for col, file_data in zip([col1, col2], st.session_state.uploaded_files):
        with col:
            params = file_data['measparams']
            html = f"**{file_data['name']}**<br>"
            html += "<br>".join(f"<b>{k}:</b> {v}" for k, v in params.items())
            st.markdown(html, unsafe_allow_html=True)


def main():
    """Main application logic."""
    init_session_state()
    setup_plot_style()
    
    # Render sidebar and get selected order
    with st.sidebar:
        order = render_sidebar()
    
    motd = '''## Suggested experiment parameters

    For best comparison between measurements we suggest to use the following parameters.

    In-contact tapping amplitude: 80 nm
    Number of acquisitions: 16
    Integration time: 20 ms
    Spectral resolution: 6 cm⁻¹
    '''
          # Hide motd once plot is successfully rendered

    motd_box = None
    if st.session_state.show_motd:
        motd_box = st.markdown(motd)

    # Main content
    if len(st.session_state.uploaded_files) == MAX_FILES and order and st.session_state.default_snr_valid:
        try:
            file_1 = st.session_state.uploaded_files[0]
            file_2 = st.session_state.uploaded_files[1]

            wn1 = file_1['data']["Wavenumber"]
            sp1 = file_1['data'][order]
            wn2 = file_2['data']["Wavenumber"]
            sp2 = file_2['data'][order]

            t_compute = time.perf_counter()
            compute_plot_data(wn1, sp1, wn2, sp2)
            st.session_state.perf_stats["compute_cached"] = time.perf_counter() - t_compute

            t_plot = time.perf_counter()
            fig = create_comparison_plot(
                file_1,
                file_2,
                order
            )
            st.pyplot(fig, width="stretch")
            st.session_state.perf_stats["plot_render"] = time.perf_counter() - t_plot
            plt.close(fig)  # Free memory

            if st.session_state.custom_snr_ranges:
                t_custom_plot = time.perf_counter()
                st.markdown("### Additional SNR graphs")
                custom_fig = create_custom_snr_plot(
                    file_1,
                    file_2,
                    order,
                    st.session_state.custom_snr_ranges
                )
                st.pyplot(custom_fig, width="stretch")
                st.session_state.perf_stats["custom_plot_render"] = time.perf_counter() - t_custom_plot
                plt.close(custom_fig)
            
            st.session_state.show_motd = False
            if motd_box:
                motd_box.empty()

            render_metadata()

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)
    elif len(st.session_state.uploaded_files) == MAX_FILES and order and not st.session_state.default_snr_valid:
        st.warning("Cannot render main SNR graphs because preset SNR ranges are not valid for these files.")


if __name__ == "__main__":
    main()
