import paramiko
import time


class DUTClient:
    def __init__(self, hostname, username, root_access=False, key_filename=None, backup_ip=None):
        self.hostname = hostname
        self.username = username
        self.backup_ip = backup_ip
        self.root_access = root_access
        self.key_filename = key_filename

        self.user = None
        self.root = None
        self.connected_host = None

    def _connect_single(self, hostname, username, timeout):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=hostname,
            username=username,
            key_filename=self.key_filename,
            timeout=timeout
        )
        return client

    def connect(self, timeout=20):
        targets = [self.hostname]
        if self.backup_ip and self.backup_ip != self.hostname:
            targets.append(self.backup_ip)

        last_exec = None
        for target in targets:
            try:
                self.user = self._connect_single(
                    hostname=target,
                    username=self.username,
                    timeout=timeout
                )

                if self.root_access:
                    self.root = self._connect_single(
                        hostname=target,
                        username="root",
                        timeout=timeout
                    )
                self.connected_host = target
                return
            except Exception as e:
                last_exec = e

                if self.user is not None:
                    self.user.close()
                    self.user = None

                if self.root is not None:
                    self.root.close()
                    self.root = None
        raise ConnectionError(
            f"Failed to connect to both primary host '{self.hostname}"
            + (f"and backup IP {self.backup_ip}" if self.backup_ip else "")

        )

    def run(self, cmd, as_root=False, async_run=False):

        client = self.root if as_root else self.user
        if client is None:
            raise RuntimeError("SSH client not connected")

        stdin, stdout, stderr = client.exec_command(cmd)

        if async_run:
            # return immediately without draining pipes
            return None, None, None

        # normal synchronous behavior
        code = stdout.channel.recv_exit_status()

        out = stdout.read().decode()
        err = stderr.read().decode()

        return code, out, err

    def reboot(self):
        try:
            self.run("reboot", as_root=True)
        except Exception:
            # SSH disconect during reboot is expected
            pass

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

    def fetch_directory(self, remote_dir, local_dir):
        if self.root is not None:
            sftp = self.root.open_sftp()
        else: 
            sftp = self.user.open_sftp()

        for entry in sftp.listdir(remote_dir):

            remote_path = f"{remote_dir}/{entry}"
            local_path = f"{local_dir}/{entry}"

            sftp.get(remote_path, local_path)

        sftp.close()