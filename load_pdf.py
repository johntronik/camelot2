import os
import sys
import warnings
import numpy as np
import pandas as pd
import copy
import cv2
import pikepdf
import fitz

from PyPDF2 import PdfFileReader
from camelot.handlers import PDFHandler
from camelot.parsers.lattice import Lattice
from camelot.parsers import Stream

from camelot.utils import (
    TemporaryDirectory,
    scale_image,
    scale_pdf,
    is_url,
    download_url,
    validate_input,
    remove_extra,
    get_table_index,
    compute_accuracy,
    compute_whitespace,
)

from camelot.image_processing import (
    adaptive_threshold,
    find_lines,
    find_contours,
    find_joints,
)

from camelot.core import (
    TableList,
    Table,
)

# ref: https://needtec.sakura.ne.jp/wod07672/2020/05/03/camelot%E3%81%A7%E7%82%B9%E7%B7%9A%E3%82%92%E5%AE%9F%E7%B7%9A%E3%81%A8%E3%81%97%E3%81%A6%E5%87%A6%E7%90%86%E3%81%99%E3%82%8B/


def image_proc_tate_tensen(threshold):
    el = np.zeros((5, 5), np.uint8)
    el[:, 1] = 1
    threshold = cv2.dilate(threshold, el, iterations=1)
    threshold = cv2.erode(threshold, el, iterations=1)
    return threshold


def image_proc_yoko_tensen(threshold):
    el = np.zeros((5, 5), np.uint8)
    el[2, :] = 1
    threshold = cv2.dilate(threshold, el, iterations=1)
    threshold = cv2.erode(threshold, el, iterations=1)
    return threshold


def image_proc_both_tensen(threshold):
    el = np.zeros((5, 5), np.uint8)
    el[2, :] = 1
    el[:, 1] = 1
    threshold = cv2.dilate(threshold, el, iterations=1)
    threshold = cv2.erode(threshold, el, iterations=1)
    return threshold


class Lattice2(Lattice):
    """
    image_proc使う用
    """

    def __init__(
        self,
        table_regions=None,
        table_areas=None,
        process_background=False,
        line_scale=15,
        copy_text=None,
        shift_text=["l", "t"],
        split_text=False,
        flag_size=False,
        strip_text="",
        line_tol=2,
        joint_tol=2,
        threshold_blocksize=15,
        threshold_constant=-2,
        iterations=0,
        resolution=300,
        backend="ghostscript",
        image_proc=None,
        **kwargs,
    ):
        self.table_regions = table_regions
        self.table_areas = table_areas
        self.process_background = process_background
        self.line_scale = line_scale
        self.copy_text = copy_text
        self.shift_text = shift_text
        self.split_text = split_text
        self.flag_size = flag_size
        self.strip_text = strip_text
        self.line_tol = line_tol
        self.joint_tol = joint_tol
        self.threshold_blocksize = threshold_blocksize
        self.threshold_constant = threshold_constant
        self.iterations = iterations
        self.resolution = resolution
        self.backend = Lattice._get_backend(backend)
        self.image_proc = image_proc

    def _generate_table_bbox(self):
        def scale_areas(areas):
            scaled_areas = []
            for area in areas:
                x1, y1, x2, y2 = area.split(",")
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                x1, y1, x2, y2 = scale_pdf((x1, y1, x2, y2), image_scalers)
                scaled_areas.append((x1, y1, abs(x2 - x1), abs(y2 - y1)))
            return scaled_areas

        self.image, self.threshold = adaptive_threshold(
            self.imagename,
            process_background=self.process_background,
            blocksize=self.threshold_blocksize,
            c=self.threshold_constant,
        )

        image_width = self.image.shape[1]
        image_height = self.image.shape[0]
        image_width_scaler = image_width / float(self.pdf_width)
        image_height_scaler = image_height / float(self.pdf_height)
        pdf_width_scaler = self.pdf_width / float(image_width)
        pdf_height_scaler = self.pdf_height / float(image_height)
        image_scalers = (image_width_scaler,
                         image_height_scaler, self.pdf_height)
        pdf_scalers = (pdf_width_scaler, pdf_height_scaler, image_height)

        ############
        # ここだけベースクラスと異なる処理
        if self.image_proc == "tate":
            self.threshold = image_proc_tate_tensen(self.threshold)
        elif self.image_proc == "yoko":
            self.threshold = image_proc_yoko_tensen(self.threshold)
        elif self.image_proc == 'both':
            self.threshold = image_proc_tate_tensen(self.threshold)
            self.threshold = image_proc_yoko_tensen(self.threshold)

        ############

        if self.table_areas is None:
            regions = None
            if self.table_regions is not None:
                regions = scale_areas(self.table_regions)

            vertical_mask, vertical_segments = find_lines(
                self.threshold,
                regions=regions,
                direction="vertical",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )
            horizontal_mask, horizontal_segments = find_lines(
                self.threshold,
                regions=regions,
                direction="horizontal",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )

            contours = find_contours(vertical_mask, horizontal_mask)
            table_bbox = find_joints(contours, vertical_mask, horizontal_mask)
        else:
            vertical_mask, vertical_segments = find_lines(
                self.threshold,
                direction="vertical",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )
            horizontal_mask, horizontal_segments = find_lines(
                self.threshold,
                direction="horizontal",
                line_scale=self.line_scale,
                iterations=self.iterations,
            )

            areas = scale_areas(self.table_areas)
            table_bbox = find_joints(areas, vertical_mask, horizontal_mask)

        self.table_bbox_unscaled = copy.deepcopy(table_bbox)

        self.table_bbox, self.vertical_segments, self.horizontal_segments = scale_image(
            table_bbox, vertical_segments, horizontal_segments, pdf_scalers
        )

    # IndexErrorで追加した部分↓↓
    def _reduce_index(t, idx, shift_text):
        """Reduces index of a text object if it lies within a spanning
        cell.

        Parameters
        ----------
        table : camelot.core.Table
        idx : list
            List of tuples of the form (r_idx, c_idx, text).
        shift_text : list
            {'l', 'r', 't', 'b'}
            Select one or more strings from above and pass them as a
            list to specify where the text in a spanning cell should
            flow.

        Returns
        -------
        indices : list
            List of tuples of the form (r_idx, c_idx, text) where
            r_idx and c_idx are new row and column indices for text.

        """
        indices = []
        for r_idx, c_idx, text in idx:
            for d in shift_text:
                if d == "l":
                    # for IndexError
                    if len(t.cells[r_idx]) <= c_idx:
                        c_idx = len(t.cells[r_idx]) - 1
                    if t.cells[r_idx][c_idx].hspan:
                        while not t.cells[r_idx][c_idx].left:
                            c_idx -= 1
                if d == "r":
                    if t.cells[r_idx][c_idx].hspan:
                        while not t.cells[r_idx][c_idx].right:
                            c_idx += 1
                if d == "t":
                    if t.cells[r_idx][c_idx].vspan:
                        while not t.cells[r_idx][c_idx].top:
                            r_idx -= 1
                if d == "b":
                    if t.cells[r_idx][c_idx].vspan:
                        while not t.cells[r_idx][c_idx].bottom:
                            r_idx += 1
            indices.append((r_idx, c_idx, text))
        return indices

    def _generate_table(self, table_idx, cols, rows, **kwargs):
        v_s = kwargs.get("v_s")
        h_s = kwargs.get("h_s")
        if v_s is None or h_s is None:
            raise ValueError("No segments found on {}".format(self.rootname))

        table = Table(cols, rows)
        # set table edges to True using ver+hor lines
        table = table.set_edges(v_s, h_s, joint_tol=self.joint_tol)
        # set table border edges to True
        table = table.set_border()
        # set spanning cells to True
        table = table.set_span()

        pos_errors = []
        # TODO: have a single list in place of two directional ones?
        # sorted on x-coordinate based on reading order i.e. LTR or RTL
        for direction in ["vertical", "horizontal"]:
            for t in self.t_bbox[direction]:
                indices, error = get_table_index(
                    table,
                    t,
                    direction,
                    split_text=self.split_text,
                    flag_size=self.flag_size,
                    strip_text=self.strip_text,
                )
                if indices[:2] != (-1, -1):
                    pos_errors.append(error)
                    indices = Lattice2._reduce_index(
                        table, indices, shift_text=self.shift_text
                    )
                    for r_idx, c_idx, text in indices:
                        table.cells[r_idx][c_idx].text = text
        accuracy = compute_accuracy([[100, pos_errors]])

        if self.copy_text is not None:
            table = Lattice._copy_spanning_text(
                table, copy_text=self.copy_text)

        data = table.data
        table.df = pd.DataFrame(data)
        table.shape = table.df.shape

        whitespace = compute_whitespace(data)
        table.flavor = "lattice"
        table.accuracy = accuracy
        table.whitespace = whitespace
        table.order = table_idx + 1
        table.page = int(os.path.basename(self.rootname).replace("page-", ""))

        # for plotting
        _text = []
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.horizontal_text])
        _text.extend([(t.x0, t.y0, t.x1, t.y1) for t in self.vertical_text])
        table._text = _text
        table._image = (self.image, self.table_bbox_unscaled)
        table._segments = (self.vertical_segments, self.horizontal_segments)
        table._textedges = None

        return table


class PDFHandler2(PDFHandler):
    def __init__(self, filepath, pages="1", password=None):
        if is_url(filepath):
            filepath = download_url(filepath)
        self.filepath = filepath
        if not filepath.lower().endswith(".pdf"):
            raise NotImplementedError("File format not supported")

        if password is None:
            self.password = ""
        else:
            self.password = password
            if sys.version_info[0] < 3:
                self.password = self.password.encode("ascii")
        self.pages = self._get_pages(self.filepath, pages)

    def _get_pages(self, filepath, pages):
        page_numbers = []
        if pages == "1":
            page_numbers.append({"start": 1, "end": 1})
        else:
            infile = PdfFileReader(open(filepath, "rb"), strict=False)
            if infile.isEncrypted:
                infile.decrypt(self.password)
            if pages == "all":
                page_numbers.append({"start": 1, "end": infile.getNumPages()})
            else:
                for r in pages.split(","):
                    if "-" in r:
                        a, b = r.split("-")
                        if b == "end":
                            b = infile.getNumPages()
                        page_numbers.append({"start": int(a), "end": int(b)})
                    else:
                        page_numbers.append({"start": int(r), "end": int(r)})
        P = []
        for p in page_numbers:
            P.extend(range(p["start"], p["end"] + 1))
        return sorted(set(P))

    def parse(
        self, flavor="lattice", suppress_stdout=False, layout_kwargs={}, **kwargs
    ):
        tables = []
        with TemporaryDirectory() as tempdir:
            for p in self.pages:
                self._save_page(self.filepath, p, tempdir)
            pages = [
                os.path.join(tempdir, "page-{0}.pdf".format(p)) for p in self.pages
            ]
            # Lattice -> Lattice2
            parser = Lattice2(
                **kwargs) if flavor == "lattice" else Stream(**kwargs)
            for p in pages:
                t = parser.extract_tables(
                    p, suppress_stdout=suppress_stdout, layout_kwargs=layout_kwargs
                )
                tables.extend(t)
        return TableList(sorted(tables))


def read_pdf(
    filepath,
    pages="1",
    password=None,
    flavor="lattice",
    suppress_stdout=False,
    layout_kwargs={},
    **kwargs
):
    if flavor not in ["lattice", "stream"]:
        raise NotImplementedError(
            "Unknown flavor specified." " Use either 'lattice' or 'stream'"
        )

    with warnings.catch_warnings():
        if suppress_stdout:
            warnings.simplefilter("ignore")

        validate_input(kwargs, flavor=flavor)
        # PDFHandler -> PDFHandler2
        p = PDFHandler2(filepath, pages=pages, password=password)
        kwargs = remove_extra(kwargs, flavor=flavor)
        tables = p.parse(
            flavor=flavor,
            suppress_stdout=suppress_stdout,
            layout_kwargs=layout_kwargs,
            **kwargs
        )
        return tables

# --- #

def img_replace(page, xref, filename=None, stream=None, pixmap=None):
    """Replace image identified by xref.
    Args:
        page: a fitz.Page object
        xref: cross reference number of image to replace
        filename, stream, pixmap: must be given as for
        page.insert_image().
    """
    if bool(filename) + bool(stream) + bool(pixmap) != 1:
        raise ValueError("Exactly one of filename/stream/pixmap must be given")
    doc = page.parent  # the owning document
    # insert new image anywhere in page
    new_xref = page.insert_image(
        page.rect, filename=filename, stream=stream, pixmap=pixmap
    )
    doc.xref_copy(new_xref, xref)  # copy over new to old
    last_contents_xref = page.get_contents()[-1]
    # new image insertion has created a new /Contents source,
    # which we will set to spaces now
    doc.update_stream(last_contents_xref, b" ")


def remove_images(filepath) -> str:
    pdf = fitz.open(filepath)
    for i in range(pdf.page_count):
        page = pdf[i]
        for image in page.get_images():
            xref, *_ = image
            pix = fitz.Pixmap(fitz.csGRAY, (0, 0, 1, 1), 1)
            pix.clear_with()  # clear all samples bytes to 0x00
            img_replace(page, xref, pixmap=pix)
        page.apply_redactions()
    path_wo_images = filepath[:-4] + '_noimage.pdf'
    pdf.save(path_wo_images, deflate=True)
    return path_wo_images


def load(filepath, options):
    pdf = pikepdf.open(filepath, allow_overwriting_input=True)
    pdf.save(filepath)

    if options['remove_images']:
        filepath = remove_images(filepath)
    print(filepath)
    tables = read_pdf(filepath, **options["camelot"])
    return tables

# ---


try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    _HAS_MPL = False
else:
    _HAS_MPL = True


class PlotMethods(object):
    def __call__(self, table, kind="text", filename=None):
        """Plot elements found on PDF page based on kind
        specified, useful for debugging and playing with different
        parameters to get the best output.
        Parameters
        ----------
        table: camelot.core.Table
            A Camelot Table.
        kind : str, optional (default: 'text')
            {'text', 'grid', 'contour', 'joint', 'line'}
            The element type for which a plot should be generated.
        filepath: str, optional (default: None)
            Absolute path for saving the generated plot.
        Returns
        -------
        fig : matplotlib.fig.Figure
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for plotting.")

        if table.flavor == "lattice" and kind in ["textedge"]:
            raise NotImplementedError(
                "Lattice flavor does not support kind='{}'".format(kind)
            )
        elif table.flavor == "stream" and kind in ["joint", "line"]:
            raise NotImplementedError(
                "Stream flavor does not support kind='{}'".format(kind)
            )

        plot_method = getattr(self, kind)
        return plot_method(table)

    def text(self, table, figsize=(12, 12)):
        """Generates a plot for all text elements present
        on the PDF page.
        Parameters
        ----------
        table : camelot.core.Table
        Returns
        -------
        fig : matplotlib.fig.Figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)  # , aspect="equal")
        xs, ys = [], []
        for t in table._text:
            xs.extend([t[0], t[2]])
            ys.extend([t[1], t[3]])
            ax.add_patch(patches.Rectangle(
                (t[0], t[1]), t[2] - t[0], t[3] - t[1]))
        ax.set_xlim(min(xs) - 10, max(xs) + 10)
        ax.set_ylim(min(ys) - 10, max(ys) + 10)
        return fig

    def grid(self, table, figsize=(12, 12)):
        """Generates a plot for the detected table grids
        on the PDF page.
        Parameters
        ----------
        table : camelot.core.Table
        Returns
        -------
        fig : matplotlib.fig.Figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)  # , aspect="equal")
        for row in table.cells:
            for cell in row:
                if cell.left:
                    ax.plot([cell.lb[0], cell.lt[0]], [cell.lb[1], cell.lt[1]])
                if cell.right:
                    ax.plot([cell.rb[0], cell.rt[0]], [cell.rb[1], cell.rt[1]])
                if cell.top:
                    ax.plot([cell.lt[0], cell.rt[0]], [cell.lt[1], cell.rt[1]])
                if cell.bottom:
                    ax.plot([cell.lb[0], cell.rb[0]], [cell.lb[1], cell.rb[1]])
        return fig

    def contour(self, table, figsize=(12, 12)):
        """Generates a plot for all table boundaries present
        on the PDF page.
        Parameters
        ----------
        table : camelot.core.Table
        Returns
        -------
        fig : matplotlib.fig.Figure
        """
        try:
            img, table_bbox = table._image
            _FOR_LATTICE = True
        except TypeError:
            img, table_bbox = (None, {table._bbox: None})
            _FOR_LATTICE = False
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)  # , aspect="equal")

        xs, ys = [], []
        if not _FOR_LATTICE:
            for t in table._text:
                xs.extend([t[0], t[2]])
                ys.extend([t[1], t[3]])
                ax.add_patch(
                    patches.Rectangle(
                        (t[0], t[1]), t[2] - t[0], t[3] - t[1], color="blue"
                    )
                )

        for t in table_bbox.keys():
            ax.add_patch(
                patches.Rectangle(
                    (t[0], t[1]), t[2] - t[0], t[3] - t[1], fill=False, color="red"
                )
            )
            if not _FOR_LATTICE:
                xs.extend([t[0], t[2]])
                ys.extend([t[1], t[3]])
                ax.set_xlim(min(xs) - 10, max(xs) + 10)
                ax.set_ylim(min(ys) - 10, max(ys) + 10)

        if _FOR_LATTICE:
            ax.imshow(img)
        return fig

    def textedge(self, table, figsize=(12, 12)):
        """Generates a plot for relevant textedges.
        Parameters
        ----------
        table : camelot.core.Table
        Returns
        -------
        fig : matplotlib.fig.Figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)  # , aspect="equal")
        xs, ys = [], []
        for t in table._text:
            xs.extend([t[0], t[2]])
            ys.extend([t[1], t[3]])
            ax.add_patch(
                patches.Rectangle((t[0], t[1]), t[2] - t[0],
                                  t[3] - t[1], color="blue")
            )
        ax.set_xlim(min(xs) - 10, max(xs) + 10)
        ax.set_ylim(min(ys) - 10, max(ys) + 10)

        for te in table._textedges:
            ax.plot([te.x, te.x], [te.y0, te.y1])

        return fig

    def joint(self, table, figsize=(12, 12)):
        """Generates a plot for all line intersections present
        on the PDF page.
        Parameters
        ----------
        table : camelot.core.Table
        Returns
        -------
        fig : matplotlib.fig.Figure
        """
        img, table_bbox = table._image
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)  # , aspect="equal")
        x_coord = []
        y_coord = []
        for k in table_bbox.keys():
            for coord in table_bbox[k]:
                x_coord.append(coord[0])
                y_coord.append(coord[1])
        ax.plot(x_coord, y_coord, "ro", markersize=16)
        ax.imshow(img)
        return fig

    def line(self, table, figsize=(12, 12)):
        """Generates a plot for all line segments present
        on the PDF page.
        Parameters
        ----------
        table : camelot.core.Table
        Returns
        -------
        fig : matplotlib.fig.Figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)  # , aspect="equal")
        vertical, horizontal = table._segments
        for v in vertical:
            ax.plot([v[0], v[2]], [v[1], v[3]])
        for h in horizontal:
            ax.plot([h[0], h[2]], [h[1], h[3]])
        return fig
