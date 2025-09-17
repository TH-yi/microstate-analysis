import os
import json


class MetricsBase():
    def ensure_dir(self, dir):
        directory = os.path.dirname(dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.logger.log_warning("Dir not exist, created:", str(directory))

    # --- public API ---
    def dump_to_json(self, json_data, output_dir, file_name):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.log_info(f"Output directory {output_dir} created.")

            if not file_name.lower().endswith(".json"):
                file_name = f"{file_name}.json"

            json_file_path = os.path.join(output_dir, file_name)

            # Sanitize before dumping
            coercions: List[Tuple[str, str, str]] = []
            safe_data = self._json_safe(json_data, _conversions=coercions)

            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(
                    safe_data,
                    json_file,
                    ensure_ascii=False,
                    indent=4,
                    separators=(',', ':'),
                    sort_keys=True
                )

            if coercions:
                # Log a compact preview of coercions
                preview = ", ".join([f"{p}:{t}->{a}" for p, t, a in coercions[:5]])
                more = f" (+{len(coercions) - 5} more)" if len(coercions) > 5 else ""
                self.logger.log_warning(f"JSON coercions applied: {preview}{more}")

            self.logger.log_info(f"Successfully dumped JSON data to {json_file_path}")

        except Exception as e:
            self.logger.log_error(f"Failed to dump JSON data: {str(e)}")

    def dump_to_json_path(self, json_data, json_file_path):
        try:
            output_dir = os.path.dirname(json_file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.log_info(f"Output directory {output_dir} created.")

            # Sanitize before dumping
            coercions: List[Tuple[str, str, str]] = []
            safe_data = self._json_safe(json_data, _conversions=coercions)

            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(
                    safe_data,
                    json_file,
                    ensure_ascii=False,
                    indent=4,
                    separators=(',', ':'),
                    sort_keys=True
                )

            if coercions:
                preview = ", ".join([f"{p}:{t}->{a}" for p, t, a in coercions[:5]])
                more = f" (+{len(coercions) - 5} more)" if len(coercions) > 5 else ""
                self.logger.log_warning(f"JSON coercions applied: {preview}{more}")

            self.logger.log_info(f"Successfully dumped JSON data to {json_file_path}")

        except Exception as e:
            self.logger.log_error(f"Failed to dump JSON data to {json_file_path}: {str(e)}")

    # --- legacy helper kept for compatibility ---
    @staticmethod
    def _to_list(x):
        """Prefer list-like conversion for arrays, otherwise return as-is."""
        try:
            import cupy as cp
            if isinstance(x, cp.ndarray):
                x = cp.asnumpy(x)
        except Exception:
            pass
        try:
            import numpy as np
            return x.tolist() if isinstance(x, np.ndarray) else x
        except Exception:
            return x

    # --- core: recursively make object JSON-safe ---
    def _json_safe(
            self,
            obj,
            _stack=None,  # recursion stack (ids currently in the path)
            _path="$",
            _conversions=None
    ):
        """
        Convert arbitrary Python objects into JSON-serializable ones.

        Fixes:
        - Only perform cycle detection on container-like objects (dict/list/tuple/set/...).
        - Use a *recursion stack* instead of a global visited set, so shared substructures
          are not misclassified as cycles.
        """
        import math, dataclasses, datetime, decimal, uuid
        from collections import deque
        from pathlib import Path

        if _stack is None:
            _stack = set()
        if _conversions is None:
            _conversions = []

        # ---------- Fast path: primitives & simple non-containers (no cycle risk) ----------
        if obj is None or isinstance(obj, (str, bool, int)):
            return obj

        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                _conversions.append((_path, "float", "nan/inf->null"))
                return None
            return obj

        if dataclasses.is_dataclass(obj):
            _conversions.append((_path, type(obj).__name__, "dataclass->dict"))
            try:
                data = dataclasses.asdict(obj)
            except Exception:
                data = {k: getattr(obj, k) for k in getattr(obj, "__dataclass_fields__", {})}
            return self._json_safe(data, _stack, _path, _conversions)

        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            _conversions.append((_path, type(obj).__name__, "datetime->iso"))
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)

        if isinstance(obj, decimal.Decimal):
            _conversions.append((_path, "Decimal", "decimal->float"))
            try:
                return float(obj)
            except Exception:
                _conversions.append((_path, "Decimal", "decimal->str"))
                return str(obj)

        if isinstance(obj, (uuid.UUID, Path)):
            _conversions.append((_path, type(obj).__name__, "->str"))
            return str(obj)

        if isinstance(obj, (bytes, bytearray, memoryview)):
            import base64
            _conversions.append((_path, type(obj).__name__, "bytes->base64"))
            return base64.b64encode(bytes(obj)).decode("ascii")

        # NumPy scalar
        try:
            import numpy as np
            if isinstance(obj, np.generic):
                _conversions.append((_path, type(obj).__name__, "np.scalar->py.scalar"))
                return obj.item()
        except Exception:
            pass

        # CuPy array -> list
        try:
            import cupy as cp
            if isinstance(obj, cp.ndarray):
                _conversions.append((_path, "cp.ndarray", "cp->np->tolist()"))
                return cp.asnumpy(obj).tolist()
        except Exception:
            pass

        # PyTorch
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                _conversions.append((_path, "torch.Tensor", "tensor->cpu->tolist()"))
                return obj.detach().cpu().tolist()
            if isinstance(obj, torch.device):
                _conversions.append((_path, "torch.device", "->str"))
                return str(obj)
        except Exception:
            pass

        # pandas
        try:
            import pandas as pd
            if isinstance(obj, (pd.Series, pd.Index)):
                _conversions.append((_path, type(obj).__name__, "tolist()"))
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                _conversions.append((_path, "pd.DataFrame", "values->list + columns"))
                return {"__dataframe__": True,
                        "columns": obj.columns.tolist(),
                        "data": obj.values.tolist()}
        except Exception:
            pass

        # NumPy ndarray
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                _conversions.append((_path, "np.ndarray", "tolist()"))
                return obj.tolist()
        except Exception:
            pass

        # Generic ".tolist()"
        tolist = getattr(obj, "tolist", None)
        if callable(tolist):
            try:
                _conversions.append((_path, type(obj).__name__, "tolist()"))
                return tolist()
            except Exception:
                pass

        # ---------- Containers (cycle detection applies only here) ----------
        is_container = isinstance(obj, (dict, list, tuple, set, frozenset, deque, range))
        if is_container:
            oid = id(obj)
            if oid in _stack:
                _conversions.append((_path, type(obj).__name__, "cycle->'<CYCLE>'"))
                return "<CYCLE>"
            _stack.add(oid)
            try:
                if isinstance(obj, dict):
                    out = {}
                    for k, v in obj.items():
                        key_path = f"{_path}.{k}" if isinstance(k, (str, int)) else f"{_path}.[key]"
                        if not isinstance(k, str):
                            _conversions.append((f"{_path}.[key]", type(k).__name__, "key->str"))
                            ks = str(self._json_safe(k, _stack, f"{_path}.[key]", _conversions))
                        else:
                            ks = k
                        out[ks] = self._json_safe(v, _stack, key_path, _conversions)
                    return out
                else:
                    lst = list(obj)
                    return [self._json_safe(v, _stack, f"{_path}[{i}]", _conversions) for i, v in enumerate(lst)]
            finally:
                _stack.remove(oid)

        # Generic iterable -> list (no cycle detection; if it recurses, it will pass above)
        if hasattr(obj, "__iter__"):
            try:
                lst = list(obj)
                _conversions.append((_path, type(obj).__name__, "iterable->list"))
                return [self._json_safe(v, _stack, f"{_path}[{i}]", _conversions) for i, v in enumerate(lst)]
            except Exception:
                pass

        # Fallback
        _conversions.append((_path, type(obj).__name__, "->str"))
        return str(obj)

