#!/usr/bin/env python3
# encoding: utf-8
"""Dot fractals generator."""

import math
import sys


class FractalConfigError(Exception):
    """Fractal configuration error."""


class FractalPattern:
    """Fractal pattern."""

    def __init__(self, points, seed1, seed2, arcs):
        """Instanciate pattern."""
        self.points = [complex(p) for p in points]
        self.seed1 = int(seed1)
        self.seed2 = int(seed2)
        self.arcs = [tuple(a) for a in arcs]
        self.scale = 1.0
        self._norm()

    def _norm(self):
        """Normalize given pattern.

        so that seed1 is (0, 0) and seed2 (1, 0).
        """
        seed1 = self.points[self.seed1]
        seed2 = self.points[self.seed2]
        if seed1 == 0j and seed2 == 1 + 0j:
            # Already normalized.
            return
        # First, translate all points so that seed1 is (0, 0)
        for i, p in enumerate(self.points):
            self.points[i] -= seed1
        seed2 -= seed1
        # Next, rotate and scale down all points so that seed2 = (1, 0).
        len_seed2 = abs(seed2)
        theta = math.acos(seed2.real / len_seed2)
        if seed2.imag < 0:
            theta = -theta
        self.scale, matrix = self.transform_matrix(-theta, 1.0 / len_seed2)
        for i, p in enumerate(self.points):
            self.points[i] = self.apply_transform(matrix, p)

    def transform_matrix_from_seeds(self, s1, s2):
        """Get a transformation matrix given a seed.

        The transformation is from unit vector 1+0j to seeds vector s2 - s1.
        """
        ds = s2 - s1
        len_s = abs(ds)
        theta = math.acos(ds.real / len_s)
        if ds.imag < 0:
            theta = -theta
        return self.transform_matrix(theta, len_s, dest=s1)

    def transform_matrix(self, theta, scale=1.0, origin=0j, dest=None):
        """Compute 2d transformation matrix."""
        if dest is None:
            dest = origin
        scaled_cos = scale * math.cos(theta)
        scaled_sin = scale * math.sin(theta)
        tx = -origin.real
        ty = -origin.imag
        t_x = dest.real
        t_y = dest.imag
        return self.scale * scale, (
            scaled_cos, -scaled_sin, t_x + tx * scaled_cos - ty * scaled_sin,
            scaled_sin, scaled_cos, t_y + tx * scaled_sin + ty * scaled_cos,
            0, 0, 1,
        )

    @classmethod
    def apply_transform(cls, matrix, c):
        """Apply transformation matrix to complex number."""
        return complex(
            round(matrix[0] * c.real + matrix[1] * c.imag + matrix[2], 7),
            round(matrix[3] * c.real + matrix[4] * c.imag + matrix[5], 7),
        )

    def apply_pattern(self, arcs):
        """Apply pattern to every pairs of points between kernel_points and last_gen.

        kernel_set and last_set should be sets of complex numbers.

        If last_gen_pairs is True, Also apply pattern for pairs of last_gen points.
        If two_ways is True, Also apply pattern for pairs between last_gen and kernel_points.
        """
        points = set()
        next_arcs = set()
        for s1, s2 in arcs:
            new_points, new_arcs = self.build_pattern_points(s1, s2)
            points.update(new_points)
            next_arcs.update(new_arcs)
        return points, next_arcs

    def build_pattern_points(self, seed1, seed2):
        """Build transformed pattern points given seeds points."""
        p_scale, matrix = self.transform_matrix_from_seeds(seed1, seed2)
        tr_points = []
        arcs = []
        for i, p in enumerate(self.points):
            if i == self.seed1:
                tr_points.append(seed1)
            elif i == self.seed2:
                tr_points.append(seed2)
            else:
                tr_points.append(self.apply_transform(matrix, p))
        for arc_s, arc_e in self.arcs:
            p_s = tr_points[arc_s]
            p_e = tr_points[arc_e]
            if p_s == p_e:
                continue
            arcs.append((p_s, p_e))
        return [(p, p_scale) for i, p in enumerate(tr_points) if i != self.seed1 and i != self.seed2], arcs

    def generate(self, seeds, generations=None):
        """Build fractal pattern for given gen-0 seed points."""
        arcs = list(seeds)
        i = 0
        points = set()
        for s1, s2 in arcs:
            p_scale, matrix = self.transform_matrix_from_seeds(s1, s2)
            points.add((s1, p_scale))
            points.add((s2, p_scale))
        yield list(points)
        while True:
            i += 1
            if generations is not None and i > generations:
                break
            sys.stderr.write("\tgeneration {}: {} generator arcs\n".format(i, len(arcs)))
            new_points, arcs = self.apply_pattern(arcs)
            yield new_points

    def get_gen1_arcs(self, seeds):
        """Get the arcs of the first gen transformation."""
        arcs = list(seeds)
        new_points, gen1_arcs = self.apply_pattern(arcs)
        arcs.extend(gen1_arcs)
        return arcs

    def build_svg(self, seeds, generations,
                  color_start=(0.0, 0.0, 0.0, 1.0), color_end=(0.75, 0.75, 0.75, 1.0), background=(1.0, 1.0, 1.0, 0.0),
                  circle_radius=0.5, max_radius=0.5, min_radius=0.1, scale_radius=False,
                  draw_gen1_arcs=False, arc_width=0.1, arc_color=(1.0, 0, 0, 1.0),
                  bbox=(None, None, None, None)):
        """Build fractal pattern as a svg given gen-0 seed points."""

        def rgba_string(c):
            rgb = tuple(round(255 * x) for x in c[:3])
            if len(c) == 3:
                return "rgb({}, {}, {})".format(*rgb)
            rgb = rgb + tuple(c[3:4])
            return "rgba({}, {}, {}, {})".format(*rgb)

        def generate_rgba(steps):
            yield rgba_string(color_start)
            step = tuple(float(color_end[i] - c) / steps for i, c in enumerate(color_start))
            for s in range(1, steps + 1):
                yield rgba_string(tuple(c + s * step[j] for j, c in enumerate(color_start)))

        if bbox is not None and bbox[0] is not None:
            user_bbox = bbox
        else:
            user_bbox = None
        bbox = None
        for s1, s2 in seeds:
            s1 = s1.conjugate()
            s2 = s2.conjugate()
            if bbox is None:
                bbox = [
                    min(s1.real, s2.real),
                    min(s1.imag, s2.imag),
                    max(s1.real, s2.real),
                    max(s1.imag, s2.imag),
                ]
            else:
                bbox = [
                    min(bbox[0], s1.real, s2.real),
                    min(bbox[1], s1.imag, s2.imag),
                    max(bbox[2], s1.real, s2.real),
                    max(bbox[3], s1.imag, s2.imag),
                ]
        svg = []
        color_iter = generate_rgba(generations)
        largest_radius = 0
        if draw_gen1_arcs:
            for p1, p2 in self.get_gen1_arcs(seeds):
                p1 = p1.conjugate()
                p2 = p2.conjugate()
                line_tag = '  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{sc}" stroke-width="{sw}"/>'
                svg.append(line_tag.format(
                    x1=p1.real,
                    y1=p1.imag,
                    x2=p2.real,
                    y2=p2.imag,
                    sc=rgba_string(arc_color),
                    sw=arc_width,
                ))
        for g in self.generate(seeds, generations):
            color = next(color_iter)
            for p, p_scale in g:
                p = p.conjugate()
                if user_bbox:
                    if not (user_bbox[0] <= p.real <= user_bbox[2]):
                        continue
                    if not (user_bbox[1] <= p.imag <= user_bbox[3]):
                        continue
                bbox[0] = min(bbox[0], p.real)
                bbox[1] = min(bbox[1], p.imag)
                bbox[2] = max(bbox[2], p.real)
                bbox[3] = max(bbox[3], p.imag)
                radius = max(min_radius, min(max_radius, p_scale * circle_radius if scale_radius else circle_radius))
                largest_radius = max(largest_radius, radius)
                svg.append(
                    '  <circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'.format(x=p.real, y=p.imag, r=radius, c=color))
        if user_bbox:
            bbox = user_bbox
        else:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            border = (0.05 * max(w, h)) or (1.05 * max_radius)
            bbox[0] = math.floor(bbox[0] - border)
            bbox[1] = math.floor(bbox[1] - border)
            bbox[2] = math.ceil(bbox[2] + border)
            bbox[3] = math.ceil(bbox[3] + border)
        svg_tag = '<svg viewBox="{x} {y} {w} {h}" style="background-color:{bc}" xmlns="http://www.w3.org/2000/svg">'
        svg.insert(0, svg_tag.format(
            x=bbox[0],
            y=bbox[1],
            w=bbox[2] - bbox[0],
            h=bbox[3] - bbox[1],
            bc=rgba_string(background),
        ))
        svg.append('</svg>\n')
        return "\n".join(svg)

    default_config = dict(
        generations=4,
        color_start=(0.4, 0.4, 0.4, 1.0),
        color_end=(0.0, 0.0, 0.0, 0.5),
        background=(1.0, 1.0, 1.0, 1.0),
        circle_radius=0.5,
        max_radius=0.5,
        min_radius=0.1,
        scale_radius=False,
        draw_gen1_arcs=False,
        arc_color=(1.0, 0.0, 0.0, 0.15),
        arc_width=0.1,
        bbox=(None, None, None, None)
    )

    @classmethod
    def build_from_config(cls, config_file, output_file):
        """Build fractal svg data from given configuration file.

        svg data is written in output_file.
        """
        import ast
        config = cls.default_config.copy()
        points = []
        seed = None
        arcs = []
        seeds = []

        value_type = 'points'

        try:
            for l in config_file:
                l = l.strip()
                if not l or l.startswith('#'):
                    continue
                if l.startswith(':'):
                    k, arg = l[1:].strip().split(None, 1)
                    k = k.lower()
                    if k not in config:
                        msg = "warning: unknown configuration key {}\n".format(k)
                        sys.stderr.write(msg)
                        continue
                    v = ast.literal_eval(arg)
                    if not isinstance(v, type(config[k])):
                        msg = "Wrong value type for {}: expect {}, got {}".format(
                            k, type(config[k]).__name__, type(v).__name__)
                        raise FractalConfigError(msg)
                    config[k] = v
                elif l.lower() == '[seeds]':
                    value_type = 'seeds'
                elif l.lower() == '[points]':
                    value_type = 'points'
                elif l.lower() == '[arcs]':
                    value_type = 'arcs'
                elif l[0] == '[':
                    msg = "Unknown section '{}'".format(l[1:-1])
                    raise FractalConfigError(msg)
                else:
                    x, y = l.split()
                    if value_type == 'points':
                        points.append(complex(float(x), float(y)))
                    elif value_type == 'arcs':
                        if seed is None:
                            seed = (int(x), int(y))
                        else:
                            arcs.append((int(x), int(y)))
                    elif value_type == 'seeds':
                        seeds.append((complex(x), complex(y)))
        except (ValueError, SyntaxError) as e:
            raise FractalConfigError("Invalid configuration file: {}".format(e))

        if len(set(points)) < 3:
            msg = "Pattern definition must contain at least 3 distinct points."
            raise FractalConfigError(msg)

        if seed is None or not arcs:
            msg = "At least 2 distinct arcs required (first one used as seed)."
            raise FractalConfigError()

        l_p = len(points)
        for i1, i2 in [seed] + arcs:
            msg = "Invalid arc: arc index {} out of bound."
            if not (0 <= i1 < l_p):
                msg = msg.format(i1)
            elif not (0 <= i2 < l_p):
                msg = msg.format(i2)
            elif points[i1] == points[i2]:
                msg = "Invalid arc: start point {0} {2} and end point {1} {2} are equal".format(i1, i2, points[i2])
            else:
                msg = None
            if msg:
                raise FractalConfigError(msg)

        seed1, seed2 = seed
        if not seeds:
            seeds = [(points[seed1], points[seed2])]
        svg = cls(points, seed1, seed2, arcs).build_svg(seeds, **config)
        output_file.write(svg)


if __name__ == '__main__':
    import contextlib

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = ['-']
    for path in paths:
        try:
            if path == '-':
                in_path = '<stdin>'
                in_file = contextlib.nullcontext(sys.stdin)
                out_path = '<stdout>'
                out_file = contextlib.nullcontext(sys.stdout)
            else:
                in_path = path
                in_file = open(path, 'r')
                out_path = (path.rpartition('.')[0] or path) + '.svg'
                out_file = open(out_path, 'w')
            with in_file as in_f, out_file as out_f:
                sys.stderr.write("build {} -> {}\n".format(in_path, out_path))
                FractalPattern.build_from_config(in_f, out_file)
        except (FileNotFoundError, FractalConfigError) as e:
            sys.stderr.write("error: {}\n".format(e))
