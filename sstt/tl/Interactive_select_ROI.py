from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


class InteractiveCurveSelector:
    def __init__(
        self,
        adata,
        spatial_key: str = "spatial",
        label_key: Optional[str] = None,
        groups: Optional[Sequence] = None,
        mode: Literal["manual", "snap"] = "manual",
        side: Literal["both", "left", "right"] = "both",
        closed: bool = False,
        sample_frac: float = 0.02,
        smoothness: float = 50,
        curve_points: int = 150,
        guide_points: int = 400,
        figsize: tuple[float, float] = (10, 8),
        seed: int = 0,
        bg_size: float = 1,
        click_size: float = 5,
        bg_color: str = "lightgrey",
        show_index: bool = False,
        show_legend: bool = False,
        show_guide: bool = True,
        show_snapped: bool = True,
        show_status: bool = False,
        guide_width: float = 2,
        curve_width: float = 4,
        zoom_step: float = 0.8,
        pan_step: float = 0.15,
        verbose: bool = True,
    ) -> None:
        plt.close("all")

        self.adata = adata
        self.spatial_key = spatial_key
        self.label_key = label_key
        self.groups = None if groups is None else [str(g) for g in groups]

        self.mode = mode
        self.side = side
        self.closed = closed

        self.sample_frac = sample_frac
        self.smoothness = smoothness
        self.curve_points = curve_points
        self.guide_points = guide_points

        self.figsize = figsize
        self.seed = seed

        self.bg_size = bg_size
        self.click_size = click_size
        self.bg_color = bg_color

        self.show_index = show_index
        self.show_legend = show_legend
        self.show_guide = show_guide
        self.show_snapped = show_snapped
        self.show_status = show_status

        self.guide_width = guide_width
        self.curve_width = curve_width

        self.zoom_step = zoom_step
        self.pan_step = pan_step
        self._initial_xlim = None
        self._initial_ylim = None

        self.verbose = verbose

        self._validate_inputs()

        self.coords = np.asarray(self.adata.obsm[self.spatial_key], dtype=np.float64)
        if self.coords.ndim != 2 or self.coords.shape[1] < 2:
            raise ValueError(
                f"adata.obsm['{self.spatial_key}'] must be a 2D array "
                "with at least two columns."
            )
        self.coords = self.coords[:, :2]

        # User-clicked control points.
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.texts = []

        # Fitted results.
        self.clicked_points = None
        self.guide_curve = None
        self.snapped_coords = None
        self.snapped_indices = None
        self.final_curve = None

        self._prepare_background()
        self._init_plot()
        self._connect_events()

        plt.show()

    def on_click(self, event) -> None:
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if event.button == 1:
            self.add_point(event.xdata, event.ydata)

        elif event.button == 3:
            self.undo_point()

        self.update_plot()

    def on_key(self, event) -> None:
        if event.key == "enter":
            self.fit()

        elif event.key == "z":
            self.undo_point()
            self.update_plot()

        elif event.key == "r":
            self.reset()

        elif event.key == "escape":
            plt.close(self.fig)

        elif event.key in {"+", "="}:
            self.zoom(factor=self.zoom_step)

        elif event.key in {"-", "_"}:
            self.zoom(factor=1.0 / self.zoom_step)

        elif event.key == "left":
            self.pan(dx=-self.pan_step, dy=0.0)

        elif event.key == "right":
            self.pan(dx=self.pan_step, dy=0.0)

        elif event.key == "up":
            self.pan(dx=0.0, dy=-self.pan_step)

        elif event.key == "down":
            self.pan(dx=0.0, dy=self.pan_step)

        elif event.key == "h":
            self.reset_view()

    def add_point(self, x: float, y: float) -> None:
        self.xs.append(float(x))
        self.ys.append(float(y))

        self._clear_results()

        if self.show_index:
            txt = self.ax.text(
                x,
                y,
                str(len(self.xs)),
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="circle,pad=0.15",
                    facecolor="white",
                    alpha=0.65,
                    edgecolor="none",
                ),
            )
            self.texts.append(txt)

        self._log(f"[Add] {len(self.xs)}: x={x:.2f}, y={y:.2f}")

    def undo_point(self) -> None:
        if len(self.xs) == 0:
            return

        x = self.xs.pop()
        y = self.ys.pop()

        if len(self.texts) > 0:
            txt = self.texts.pop()
            txt.remove()

        self._clear_results()

        self._log(f"[Undo] x={x:.2f}, y={y:.2f}")

    def update_plot(self) -> None:
        if self.closed and len(self.xs) > 2:
            plot_xs = self.xs + [self.xs[0]]
            plot_ys = self.ys + [self.ys[0]]
        else:
            plot_xs = self.xs
            plot_ys = self.ys

        self.click_line.set_data(plot_xs, plot_ys)
        self._update_status()
        self.fig.canvas.draw_idle()

    def zoom(self, factor: float = 0.8) -> None:
        if factor <= 0:
            raise ValueError("`factor` must be positive.")

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_center = 0.5 * (xlim[0] + xlim[1])
        y_center = 0.5 * (ylim[0] + ylim[1])

        x_half = 0.5 * abs(xlim[1] - xlim[0]) * factor
        y_half = 0.5 * abs(ylim[1] - ylim[0]) * factor

        self.ax.set_xlim(x_center - x_half, x_center + x_half)

        if self.ax.yaxis_inverted():
            self.ax.set_ylim(y_center + y_half, y_center - y_half)
        else:
            self.ax.set_ylim(y_center - y_half, y_center + y_half)

        self.fig.canvas.draw_idle()

    def pan(self, dx: float = 0.0, dy: float = 0.0) -> None:
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        x_width = xlim[1] - xlim[0]
        y_height = ylim[1] - ylim[0]

        x_shift = dx * x_width
        y_shift = dy * y_height

        # Preserve screen-oriented panning when the y-axis is inverted.
        self.ax.set_xlim(xlim[0] + x_shift, xlim[1] + x_shift)
        self.ax.set_ylim(ylim[0] - y_shift, ylim[1] - y_shift)

        self.fig.canvas.draw_idle()

    def reset_view(self) -> None:
        if self._initial_xlim is not None:
            self.ax.set_xlim(self._initial_xlim)

        if self._initial_ylim is not None:
            self.ax.set_ylim(self._initial_ylim)

        self.fig.canvas.draw_idle()

    def fit(self) -> None:
        if len(self.xs) < 4:
            self._log("[Warning] Please click at least 4 points.")
            return

        self.clicked_points = np.column_stack((self.xs, self.ys))

        self._log(f"[Info] mode = {self.mode}")
        self._log(f"[Info] side = {self.side}")
        self._log(f"[Info] closed = {self.closed}")
        self._log(f"[Info] clicked points = {self.clicked_points.shape[0]}")

        try:
            self.guide_curve = self._fit_curve(
                self.clicked_points,
                n_points=self.guide_points,
                smoothness=self.smoothness,
                closed=self.closed,
            )

            self._log(f"[Info] guide curve points = {self.guide_curve.shape[0]}")

            if self.show_guide:
                self.guide_line.set_data(
                    self.guide_curve[:, 0],
                    self.guide_curve[:, 1],
                )
            else:
                self.guide_line.set_data([], [])

            if self.mode == "manual":
                self._log("[Info] Using manual mode.")

                self.final_curve = self._fit_curve(
                    self.clicked_points,
                    n_points=self.curve_points,
                    smoothness=self.smoothness,
                    closed=self.closed,
                )

                self.snapped_coords = None
                self.snapped_indices = None
                self.snapped_scatter.set_offsets(np.empty((0, 2)))

            elif self.mode == "snap":
                self._log("[Info] Using snap mode.")
                self._log(f"[Info] label_key = {self.label_key}")
                self._log(f"[Info] groups = {self.groups}")

                self.snapped_indices, self.snapped_coords = self._snap_to_group(
                    self.guide_curve
                )

                self._log(
                    f"[Info] snapped unique cells = {self.snapped_coords.shape[0]}"
                )

                if self.show_snapped:
                    self.snapped_scatter.set_offsets(self.snapped_coords)
                else:
                    self.snapped_scatter.set_offsets(np.empty((0, 2)))

                self.final_curve = self._fit_curve(
                    self.snapped_coords,
                    n_points=self.curve_points,
                    smoothness=self.smoothness,
                    closed=self.closed,
                )

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self.final_line.set_data(
                self.final_curve[:, 0],
                self.final_curve[:, 1],
            )

            self.update_plot()

            self._log(
                f"[Success] mode={self.mode}, side={self.side}, "
                f"closed={self.closed}. Final curve generated."
            )

        except Exception as e:
            self._log(f"[Error] Fitting failed: {e}")

            self.final_curve = None
            self.snapped_coords = None
            self.snapped_indices = None

            self.final_line.set_data([], [])
            self.snapped_scatter.set_offsets(np.empty((0, 2)))

            self.update_plot()

    def reset(self) -> None:
        self.xs = []
        self.ys = []

        for txt in self.texts:
            txt.remove()

        self.texts = []

        self.click_line.set_data([], [])
        self._clear_results()
        self._update_status()

        self.fig.canvas.draw_idle()

        self._log("[Reset] All selected points cleared.")

    def get_curve(self, as_dataframe: bool = False):
        if self.final_curve is None:
            self._log("[Warning] No curve generated yet. Draw points and press Enter.")
            return None

        result = {
            "mode": self.mode,
            "side": self.side,
            "closed": self.closed,
            "spatial_key": self.spatial_key,
            "clicked_points": self.clicked_points,
            "guide_curve": self.guide_curve,
            "final_curve": self.final_curve,
            "snapped_coords": self.snapped_coords,
            "snapped_indices": self.snapped_indices,
            "label_key": self.label_key,
            "groups": self.groups,
            "shown_indices": self.bg_idx,
            "smoothness": self.smoothness,
            "curve_points": self.curve_points,
            "guide_points": self.guide_points,
        }

        if as_dataframe:
            result["clicked_points_df"] = _points_to_dataframe(
                self.clicked_points,
                prefix="clicked",
            )
            result["guide_curve_df"] = _points_to_dataframe(
                self.guide_curve,
                prefix="guide",
            )
            result["final_curve_df"] = _points_to_dataframe(
                self.final_curve,
                prefix="final",
            )

            if self.snapped_coords is not None:
                result["snapped_coords_df"] = _points_to_dataframe(
                    self.snapped_coords,
                    prefix="snapped",
                )

        return result

    def _validate_inputs(self) -> None:
        if self.mode not in {"manual", "snap"}:
            raise ValueError("`mode` must be either 'manual' or 'snap'.")

        if self.side not in {"both", "left", "right"}:
            raise ValueError("`side` must be one of {'both', 'left', 'right'}.")

        if self.mode == "snap":
            if self.label_key is None or self.groups is None:
                raise ValueError(
                    "`mode='snap'` requires both `label_key` and `groups`."
                )

        if self.spatial_key not in self.adata.obsm:
            raise KeyError(f"adata.obsm['{self.spatial_key}'] not found.")

        if not (0 < self.sample_frac <= 1):
            raise ValueError("`sample_frac` must be in the range (0, 1].")

        if self.curve_points < 4:
            raise ValueError("`curve_points` must be >= 4.")

        if self.guide_points < 4:
            raise ValueError("`guide_points` must be >= 4.")

        if self.zoom_step <= 0:
            raise ValueError("`zoom_step` must be positive.")

        if self.pan_step <= 0:
            raise ValueError("`pan_step` must be positive.")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _prepare_background(self) -> None:
        n_cells = self.coords.shape[0]
        all_idx = np.arange(n_cells)

        if self.label_key is not None:
            if self.label_key not in self.adata.obs.columns:
                raise KeyError(
                    f"`label_key='{self.label_key}'` not found in adata.obs."
                )

            labels = self.adata.obs[self.label_key].astype(str).values

            if self.groups is not None:
                mask = np.isin(labels, self.groups)
                all_idx = all_idx[mask]

                if len(all_idx) == 0:
                    raise ValueError(
                        f"No cells found for label_key='{self.label_key}' "
                        f"with groups={self.groups}."
                    )

        n_available = len(all_idx)
        n_sample = int(n_available * self.sample_frac)
        n_sample = max(1, min(n_available, n_sample))

        rng = np.random.default_rng(self.seed)
        self.bg_idx = rng.choice(all_idx, n_sample, replace=False)
        self.bg_coords = self.coords[self.bg_idx]

    def _init_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        self._plot_background()

        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()

        self._initial_xlim = self.ax.get_xlim()
        self._initial_ylim = self.ax.get_ylim()

        title = (
            "Left click: add | Right click / z: undo | Enter: fit | "
            "r: reset | Esc: close\n"
            "+/-: zoom | Arrow keys: pan | h: reset view | "
            "left/right = visual side along first clicked point → last clicked point"
        )
        self.ax.set_title(title, fontsize=10, fontweight="bold")

        (self.click_line,) = self.ax.plot(
            [],
            [],
            "ro-",
            markersize=self.click_size,
            linewidth=1.5,
            alpha=0.9,
            label="clicked points",
        )

        (self.guide_line,) = self.ax.plot(
            [],
            [],
            color="orange",
            linewidth=self.guide_width,
            alpha=0.85,
            linestyle="--",
            label="guide curve",
        )

        self.snapped_scatter = self.ax.scatter(
            [],
            [],
            s=18,
            c="blue",
            alpha=0.85,
            edgecolors="none",
            label="snapped cells",
        )

        (self.final_line,) = self.ax.plot(
            [],
            [],
            "k-",
            linewidth=self.curve_width,
            alpha=0.95,
            label="final curve",
        )

        if self.show_status:
            self.status_text = self.ax.text(
                0.01,
                0.99,
                "",
                transform=self.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                ),
            )
        else:
            self.status_text = None

        self._update_status()

    def _plot_background(self) -> None:
        if self.label_key is not None:
            labels = self.adata.obs[self.label_key].astype(str).iloc[self.bg_idx].values

            if self.groups is not None:
                categories = self.groups
            else:
                categories = self._get_label_categories()

            color_map = self._get_label_color_map(categories)

            for cat in categories:
                cat_mask = labels == str(cat)

                if np.sum(cat_mask) > 0:
                    self.ax.scatter(
                        self.bg_coords[cat_mask, 0],
                        self.bg_coords[cat_mask, 1],
                        s=self.bg_size,
                        c=[color_map.get(str(cat), self.bg_color)],
                        label=str(cat),
                        alpha=0.6,
                        edgecolors="none",
                        rasterized=True,
                    )

            if self.show_legend:
                self.ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    markerscale=5,
                    fontsize=9,
                    title=self.label_key,
                )
                self.fig.subplots_adjust(right=0.8)

        else:
            self.ax.scatter(
                self.bg_coords[:, 0],
                self.bg_coords[:, 1],
                s=self.bg_size,
                c=self.bg_color,
                alpha=0.5,
                edgecolors="none",
                rasterized=True,
            )

    def _get_label_categories(self) -> list[str]:
        series = self.adata.obs[self.label_key]

        if pd.api.types.is_categorical_dtype(series):
            return [str(x) for x in series.cat.categories]

        return sorted(series.astype(str).unique().tolist())

    def _get_label_color_map(self, categories: Sequence[str]) -> dict[str, Any]:
        color_key = f"{self.label_key}_colors"

        if color_key in self.adata.uns:
            series = self.adata.obs[self.label_key]

            if pd.api.types.is_categorical_dtype(series):
                all_categories = [str(x) for x in series.cat.categories]
            else:
                all_categories = sorted(series.astype(str).unique().tolist())

            return dict(zip(all_categories, self.adata.uns[color_key]))

        self._log(f"[Info] adata.uns['{color_key}'] not found. Using tab20 colors.")

        cmap = plt.get_cmap("tab20")
        return {str(cat): cmap(i % 20) for i, cat in enumerate(categories)}

    def _connect_events(self) -> None:
        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_click,
        )
        self.cid_key = self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key,
        )

    def _clear_results(self) -> None:
        self.clicked_points = None
        self.guide_curve = None
        self.snapped_coords = None
        self.snapped_indices = None
        self.final_curve = None

        self.guide_line.set_data([], [])
        self.final_line.set_data([], [])
        self.snapped_scatter.set_offsets(np.empty((0, 2)))

    def _update_status(self) -> None:
        if self.status_text is None:
            return

        if self.final_curve is None:
            curve_status = "not fitted"
        else:
            curve_status = f"fitted: {self.final_curve.shape[0]} points"

        if self.snapped_coords is None:
            snap_status = "snapped: none"
        else:
            snap_status = f"snapped: {self.snapped_coords.shape[0]} cells"

        self.status_text.set_text(
            f"Mode: {self.mode}\n"
            f"Side: {self.side}\n"
            f"Closed: {self.closed}\n"
            f"Clicked points: {len(self.xs)}\n"
            f"Curve: {curve_status}\n"
            f"{snap_status}"
        )

    def _fit_curve(
        self,
        points,
        n_points: Optional[int] = None,
        smoothness: Optional[float] = None,
        closed: Optional[bool] = None,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)

        if points.shape[0] < 4:
            raise ValueError("At least 4 points are required to fit a spline curve.")

        if n_points is None:
            n_points = self.curve_points

        if smoothness is None:
            smoothness = self.smoothness

        if closed is None:
            closed = self.closed

        xs = points[:, 0]
        ys = points[:, 1]

        tck, _ = splprep(
            [xs, ys],
            s=smoothness,
            per=closed,
        )

        u_fine = np.linspace(0, 1, n_points, endpoint=not closed)
        x_fine, y_fine = splev(u_fine, tck)

        curve = np.vstack([x_fine, y_fine]).T
        return curve

    def _get_target_cells(self) -> tuple[np.ndarray, np.ndarray]:
        labels = self.adata.obs[self.label_key].astype(str).values
        mask = np.isin(labels, self.groups)

        target_idx = np.where(mask)[0]
        target_coords = self.coords[target_idx]

        if target_coords.shape[0] == 0:
            raise ValueError(
                f"No target cells found for label_key='{self.label_key}', "
                f"groups={self.groups}."
            )

        return target_idx, target_coords

    def _snap_to_group(self, guide_curve) -> tuple[np.ndarray, np.ndarray]:
        target_idx, target_coords = self._get_target_cells()

        guide_curve = np.asarray(guide_curve, dtype=np.float64)
        n_guide = guide_curve.shape[0]

        selected_local = []

        if self.side == "both":
            tree = cKDTree(target_coords)
            _, local_idx = tree.query(guide_curve, k=1)

            selected_local = [int(i) for i in local_idx]

        elif self.side in {"left", "right"}:
            for i in range(n_guide):
                p = guide_curve[i]

                if i == 0:
                    tangent = guide_curve[i + 1] - guide_curve[i]
                elif i == n_guide - 1:
                    tangent = guide_curve[i] - guide_curve[i - 1]
                else:
                    tangent = guide_curve[i + 1] - guide_curve[i - 1]

                norm = np.linalg.norm(tangent)

                if norm < 1e-8:
                    continue

                tangent = tangent / norm

                # In regular Cartesian coordinates, the left normal is [-ty, tx].
                # Since the plot uses an inverted y-axis, visual left is reversed.
                if self.ax.yaxis_inverted():
                    left_normal = np.array([tangent[1], -tangent[0]])
                else:
                    left_normal = np.array([-tangent[1], tangent[0]])

                if self.side == "left":
                    normal = left_normal
                else:
                    normal = -left_normal

                vec = target_coords - p
                side_projection = vec @ normal

                candidate_local = np.where(side_projection > 0)[0]

                if len(candidate_local) == 0:
                    continue

                candidate_vec = vec[candidate_local]
                candidate_dist = np.sqrt(np.sum(candidate_vec * candidate_vec, axis=1))

                nearest_local = candidate_local[np.argmin(candidate_dist)]
                selected_local.append(int(nearest_local))

        else:
            raise ValueError("`side` must be one of {'both', 'left', 'right'}.")

        if len(selected_local) < 4:
            raise ValueError(
                "Too few snapped cells. Try side='both', switching left/right, "
                "or drawing the guide curve closer to the target cells."
            )

        seen = set()
        ordered_local = []

        for i in selected_local:
            if i not in seen:
                seen.add(i)
                ordered_local.append(i)

        ordered_local = np.array(ordered_local, dtype=int)

        if len(ordered_local) < 4:
            raise ValueError("Too few unique snapped cells to fit a curve.")

        snapped_indices = target_idx[ordered_local]
        snapped_coords = self.coords[snapped_indices]

        return snapped_indices, snapped_coords


def _points_to_dataframe(points, prefix: str = "point") -> Optional[pd.DataFrame]:
    if points is None:
        return None

    points = np.asarray(points)

    return pd.DataFrame(
        {
            f"{prefix}_index": np.arange(points.shape[0]),
            "x": points[:, 0],
            "y": points[:, 1],
        }
    )
