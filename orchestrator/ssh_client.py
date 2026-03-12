import paramiko
import time


class DUTClient:
    def __init__(self, hostname, username, root_access=False, key_filename=None):
        self.hostname = hostname
        self.username = username
        self.root_access = root_access
        self.key_filename = key_filename

        self.user = None
        self.root = None

    def connect(self, timeout=20):
        self.user = paramiko.SSHClient()
        self.user.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self.user.connect(
            hostname=self.hostname,
            username=self.username,
            key_filename=self.key_filename,
            timeout=timeout
        )

        if self.root_access:
            self.root = paramiko.SSHClient()
            self.root.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            self.root.connect(
                hostname=self.hostname,
                username="root",
                key_filename=self.key_filename,
                timeout=timeout
            )

    def run(self, cmd, as_root=False):

        client = self.root if as_root else self.user
        if client is None:
            raise RuntimeError("SSH client not connected")

        stdin, stdout, stderr = client.exec_command(cmd)

        out = stdout.read().decode()
        err = stderr.read().decode()
        code = stdout.channel.recv_exit_status()

        return code, out, err

    def reboot(self):
        self.run("reboot", as_root=True)

    def wait_for_ssh(self, timeout_sec=300, retry_interval=5):

        deadline = time.time() + timeout_sec

        while time.time() < deadline:
            try:
                self.connect(timeout=10)
                return True
            except Exception:
                time.sleep(retry_interval)

        return False

    def close(self):
        if self.user:
            self.user.close()
        if self.root:
            self.root.close()