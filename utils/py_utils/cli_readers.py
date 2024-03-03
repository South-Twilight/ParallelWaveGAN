import io
import logging
import sys

import h5py
import kaldiio
import soundfile


def file_reader_helper(
    rspecifier: str,
    filetype: str = "mat",
    return_shape: bool = False,
    segments: str = None,
):
    """Read uttid and array in kaldi style

    This function might be a bit confusing as "ark" is used
    for HDF5 to imitate "kaldi-rspecifier".

    Args:
        rspecifier: Give as "ark:feats.ark" or "scp:feats.scp"
        filetype: "mat" is kaldi-martix, "hdf5": HDF5
        return_shape: Return the shape of the matrix,
            instead of the matrix. This can reduce IO cost for HDF5.
    Returns:
        Generator[Tuple[str, np.ndarray], None, None]:

    Examples:
        Read from kaldi-matrix ark file:

        >>> for u, array in file_reader_helper('ark:feats.ark', 'mat'):
        ...     array

        Read from HDF5 file:

        >>> for u, array in file_reader_helper('ark:feats.h5', 'hdf5'):
        ...     array

    """
    if filetype == "mat":
        return KaldiReader(rspecifier, return_shape=return_shape, segments=segments)
    elif filetype == "hdf5":
        return HDF5Reader(rspecifier, return_shape=return_shape)
    elif filetype == "sound.hdf5":
        return SoundHDF5Reader(rspecifier, return_shape=return_shape)
    elif filetype == "sound":
        return SoundReader(rspecifier, return_shape=return_shape)
    else:
        raise NotImplementedError(f"filetype={filetype}")


class KaldiReader:
    def __init__(self, rspecifier, return_shape=False, segments=None):
        self.rspecifier = rspecifier
        self.return_shape = return_shape
        self.segments = segments

    def __iter__(self):
        with kaldiio.ReadHelper(self.rspecifier, segments=self.segments) as reader:
            for key, array in reader:
                if self.return_shape:
                    array = array.shape
                yield key, array


class HDF5Reader:
    def __init__(self, rspecifier, return_shape=False):
        if ":" not in rspecifier:
            raise ValueError(
                'Give "rspecifier" such as "ark:some.ark: {}"'.format(self.rspecifier)
            )
        self.rspecifier = rspecifier
        self.ark_or_scp, self.filepath = self.rspecifier.split(":", 1)
        if self.ark_or_scp not in ["ark", "scp"]:
            raise ValueError(f"Must be scp or ark: {self.ark_or_scp}")

        self.return_shape = return_shape

    def __iter__(self):
        if self.ark_or_scp == "scp":
            hdf5_dict = {}
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    key, value = line.rstrip().split(None, 1)

                    if ":" not in value:
                        raise RuntimeError(
                            "scp file for hdf5 should be like: "
                            '"uttid filepath.h5:key": {}({})'.format(
                                line, self.filepath
                            )
                        )
                    path, h5_key = value.split(":", 1)

                    hdf5_file = hdf5_dict.get(path)
                    if hdf5_file is None:
                        try:
                            hdf5_file = h5py.File(path, "r")
                        except Exception:
                            logging.error("Error when loading {}".format(path))
                            raise
                        hdf5_dict[path] = hdf5_file

                    try:
                        data = hdf5_file[h5_key]
                    except Exception:
                        logging.error(
                            "Error when loading {} with key={}".format(path, h5_key)
                        )
                        raise

                    if self.return_shape:
                        yield key, data.shape
                    else:
                        yield key, data[()]

            # Closing all files
            for k in hdf5_dict:
                try:
                    hdf5_dict[k].close()
                except Exception:
                    pass

        else:
            if self.filepath == "-":
                # Required h5py>=2.9
                filepath = io.BytesIO(sys.stdin.buffer.read())
            else:
                filepath = self.filepath
            with h5py.File(filepath, "r") as f:
                for key in f:
                    if self.return_shape:
                        yield key, f[key].shape
                    else:
                        yield key, f[key][()]


class SoundHDF5Reader:
    def __init__(self, rspecifier, return_shape=False):
        if ":" not in rspecifier:
            raise ValueError(
                'Give "rspecifier" such as "ark:some.ark: {}"'.format(rspecifier)
            )
        self.ark_or_scp, self.filepath = rspecifier.split(":", 1)
        if self.ark_or_scp not in ["ark", "scp"]:
            raise ValueError(f"Must be scp or ark: {self.ark_or_scp}")
        self.return_shape = return_shape

    def __iter__(self):
        if self.ark_or_scp == "scp":
            hdf5_dict = {}
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    key, value = line.rstrip().split(None, 1)

                    if ":" not in value:
                        raise RuntimeError(
                            "scp file for hdf5 should be like: "
                            '"uttid filepath.h5:key": {}({})'.format(
                                line, self.filepath
                            )
                        )
                    path, h5_key = value.split(":", 1)

                    hdf5_file = hdf5_dict.get(path)
                    if hdf5_file is None:
                        try:
                            hdf5_file = SoundHDF5File(path, "r")
                        except Exception:
                            logging.error("Error when loading {}".format(path))
                            raise
                        hdf5_dict[path] = hdf5_file

                    try:
                        data = hdf5_file[h5_key]
                    except Exception:
                        logging.error(
                            "Error when loading {} with key={}".format(path, h5_key)
                        )
                        raise

                    # Change Tuple[ndarray, int] -> Tuple[int, ndarray]
                    # (soundfile style -> scipy style)
                    array, rate = data
                    if self.return_shape:
                        array = array.shape
                    yield key, (rate, array)

            # Closing all files
            for k in hdf5_dict:
                try:
                    hdf5_dict[k].close()
                except Exception:
                    pass

        else:
            if self.filepath == "-":
                # Required h5py>=2.9
                filepath = io.BytesIO(sys.stdin.buffer.read())
            else:
                filepath = self.filepath
            for key, (a, r) in SoundHDF5File(filepath, "r").items():
                if self.return_shape:
                    a = a.shape
                yield key, (r, a)


class SoundReader:
    def __init__(self, rspecifier, return_shape=False):
        if ":" not in rspecifier:
            raise ValueError(
                'Give "rspecifier" such as "scp:some.scp: {}"'.format(rspecifier)
            )
        self.ark_or_scp, self.filepath = rspecifier.split(":", 1)
        if self.ark_or_scp != "scp":
            raise ValueError(
                'Only supporting "scp" for sound file: {}'.format(self.ark_or_scp)
            )
        self.return_shape = return_shape

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                key, sound_file_path = line.rstrip().split(None, 1)
                # Assume PCM16
                array, rate = soundfile.read(sound_file_path, dtype="int16")
                # Change Tuple[ndarray, int] -> Tuple[int, ndarray]
                # (soundfile style -> scipy style)
                if self.return_shape:
                    array = array.shape
                yield key, (rate, array)


class SoundHDF5File(object):
    """Collecting sound files to a HDF5 file

    >>> f = SoundHDF5File('a.flac.h5', mode='a')
    >>> array = np.random.randint(0, 100, 100, dtype=np.int16)
    >>> f['id'] = (array, 16000)
    >>> array, rate = f['id']


    :param: str filepath:
    :param: str mode:
    :param: str format: The type used when saving wav. flac, nist, htk, etc.
    :param: str dtype:

    """

    def __init__(self, filepath, mode="r+", format=None, dtype="int16", **kwargs):
        self.filepath = filepath
        self.mode = mode
        self.dtype = dtype

        self.file = h5py.File(filepath, mode, **kwargs)
        if format is None:
            # filepath = a.flac.h5 -> format = flac
            second_ext = os.path.splitext(os.path.splitext(filepath)[0])[1]
            format = second_ext[1:]
            if format.upper() not in soundfile.available_formats():
                # If not found, flac is selected
                format = "flac"

        # This format affects only saving
        self.format = format

    def __repr__(self):
        return '<SoundHDF5 file "{}" (mode {}, format {}, type {})>'.format(
            self.filepath, self.mode, self.format, self.dtype
        )

    def create_dataset(self, name, shape=None, data=None, **kwds):
        f = io.BytesIO()
        array, rate = data
        soundfile.write(f, array, rate, format=self.format)
        self.file.create_dataset(name, shape=shape, data=np.void(f.getvalue()), **kwds)

    def __setitem__(self, name, data):
        self.create_dataset(name, data=data)

    def __getitem__(self, key):
        data = self.file[key][()]
        f = io.BytesIO(data.tobytes())
        array, rate = soundfile.read(f, dtype=self.dtype)
        return array, rate

    def keys(self):
        return self.file.keys()

    def values(self):
        for k in self.file:
            yield self[k]

    def items(self):
        for k in self.file:
            yield k, self[k]

    def __iter__(self):
        return iter(self.file)

    def __contains__(self, item):
        return item in self.file

    def __len__(self, item):
        return len(self.file)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def close(self):
        self.file.close()
